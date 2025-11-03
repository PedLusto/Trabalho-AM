import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings

# Ignorar avisos futuros do Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Iniciando script para gerar gráficos (Versão Corrigida)...")

# --- Carregar Dados ---
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("ERRO: Arquivo 'train.csv' não encontrado.")
    print("Por favor, baixe o 'train.csv' do Kaggle e coloque na mesma pasta.")
    exit()

# --- CORREÇÃO: Limpar espaços extras, caso existam ---
df.columns = df.columns.str.strip()

# --- Configurações Visuais ---
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# --- GRÁFICO 1: Histograma do SalePrice ---
print("Gerando Gráfico 1: Histograma do SalePrice...")
plt.figure()
# 'SalePrice' está correto no seu CSV
sns.histplot(df['SalePrice'], kde=True, bins=50)
plt.title('Distribuição do SalePrice (Original)', fontsize=16)
plt.xlabel('Preço de Venda (USD)', fontsize=12)
plt.ylabel('Contagem', fontsize=12)
plt.savefig('histograma_saleprice.png', bbox_inches='tight')
plt.clf() # Limpa a figura

# --- GRÁFICO 2: Dispersão SalePrice vs Gr Liv Area ---
print("Gerando Gráfico 2: Dispersão SalePrice vs Gr Liv Area...")
plt.figure()
# CORREÇÃO: Usando 'Gr Liv Area' (com espaço)
sns.scatterplot(x=df['Gr Liv Area'], y=df['SalePrice'], alpha=0.6)
plt.title('SalePrice vs. Gr Liv Area', fontsize=16)
plt.xlabel('Área de Estar (Gr Liv Area)', fontsize=12)
plt.ylabel('Preço de Venda (SalePrice)', fontsize=12)
plt.savefig('scatter_grlivarea.png', bbox_inches='tight')
plt.clf() # Limpa a figura

# --- GRÁFICO 3: Importância dos Atributos (XGBoost) ---
print("Gerando Gráfico 3: Importância dos Atributos (XGBoost)...")
print("(Isso pode levar alguns segundos, treinando o modelo...)")

# Preparar dados para o XGBoost
y = np.log1p(df['SalePrice'])
# CORREÇÃO: Removendo 'Order' e 'PID' (que são IDs) em vez de 'Id'
X = df.drop(['SalePrice', 'Order', 'PID'], axis=1)

# Preencher NaNs
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include='object').columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
    
for col in cat_cols:
    X[col] = X[col].fillna('None')
    
# Codificar categóricos (XGBoost lida bem com Label Encoding)
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Treinar modelo XGBoost (simples)
model = xgb.XGBRegressor(objective='reg:squarederror', 
                         n_estimators=500, 
                         learning_rate=0.05,
                         random_state=42,
                         n_jobs=-1,
                         # Diga ao XGBoost para lidar com nomes de colunas com espaços
                         validate_parameters=False) 
model.fit(X, y)

# Obter e processar importâncias
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Atributo': feature_names, 'Importância': importances})
importance_df = importance_df.sort_values(by='Importância', ascending=False).head(10)

# Plotar
plt.figure()
sns.barplot(x='Importância', y='Atributo', data=importance_df, palette='viridis')
plt.title('Top 10 Atributos Mais Importantes (XGBoost)', fontsize=16)
plt.xlabel('Importância Relativa', fontsize=12)
plt.ylabel('Atributo', fontsize=12)
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.clf() # Limpa a figura

print("\n--- Concluído! ---")
print("Três arquivos foram gerados:")
print("1. histograma_saleprice.png")
print("2. scatter_grlivarea.png")
print("3. feature_importance.png")
print("\nFaça o upload desses 3 arquivos para o seu projeto Overleaf.")