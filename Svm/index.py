import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Importa o SVM
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Carregar o dataset
data = pd.read_csv('data-base/smartphone_dataset_pt_br.csv')

# Remove as colunas "Modelo" e "Resolução" para testes futuros com novos smartphones
data = data.drop(columns=['Modelo', 'Resolução'])

# Imputar valores ausentes com a média para colunas numéricas
numeric_data = data.select_dtypes(include=['float64', 'int64'])
for column in numeric_data.columns:
    mean = int(data[column].mean(skipna=True))
    data[column] = data[column].replace(np.nan, mean)

# Imputar valores ausentes com a moda para colunas de string
string_data = data.select_dtypes(include=['object'])
for column in string_data:
    mode = data[column].mode()[0]
    data[column] = data[column].replace(np.nan, mode)

# Codificar dados de string para análise
label_encoders = {}
for column in string_data:
    label_encoder = LabelEncoder()
    data[column] = label_encoder.fit_transform(data[column])
    label_encoders[column] = label_encoder  # Salvar o encoder para uso posterior

# Transformar os valores de "Avaliação" em categorias
ranges = [0, 70, 85, 100]  # Definir faixas: Low (0-70), Medium (70-85), High (85-100)
labels = ['Low', 'Medium', 'High']
data['Avaliação'] = pd.cut(data['Avaliação'], bins=ranges, labels=labels)

# Remover a coluna "Avaliação" dos dados e salvar em X, e a coluna "Avaliação" em y
X = data.drop(columns=['Avaliação'])
y = data['Avaliação']

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separar 20% dos dados para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Treinar o modelo SVM com kernel linear
svm = SVC(kernel='linear')  # Pode-se alterar o kernel dependendo do problema
svm.fit(X_train, y_train)

# Fazer previsões com o modelo SVM
y_pred = svm.predict(X_test)

# Avaliar a precisão do modelo
print('Acurácia do modelo SVM:', (accuracy_score(y_test, y_pred) * 100).__round__(1), '%')

# Adicionar as previsões no dataframe de teste
df_test = X_test  # Dados de teste (características)
df_test = pd.DataFrame(df_test, columns=data.drop(columns=['Avaliação']).columns)  # Coloca os nomes das colunas

# Adicionar as previsões no dataframe
df_test['Avaliação_Prevista'] = y_pred

# Adicionar a coluna de 'Avaliação' real para comparar
df_test['Avaliação_Real'] = y_test

# Ordenar pelo valor da avaliação prevista, de preferência para 'High'
df_test['Avaliação_Prevista'] = df_test['Avaliação_Prevista'].map({'Low': 0, 'Medium': 1, 'High': 2})  # Mapear as categorias para números
df_test_sorted = df_test.sort_values(by='Avaliação_Prevista', ascending=False)

# Exibir os 10 melhores celulares com base na previsão de avaliação
print("\nTop 10 celulares com as melhores avaliações previstas:")
print(df_test_sorted.head(10))

# Função de pré-processamento para novos dados
def preprocess_new_data(new_data):
    # Codificar os dados com os encoders salvos
    for column, encoder in label_encoders.items():
        new_data[column] = encoder.transform(new_data[column])

    new_data = scaler.transform(new_data)
    return new_data

# Exemplo de novo dado (smartphone específico)
new_data = pd.DataFrame({
    'Marca': ['samsung'],
    'Preço': [4229.00],
    '5G': [0],
    'NFC': [1],
    'IR_Blaster': [0],
    'Marca_Processador': ['exynos'],
    'Qtd_Cores': [8],
    'Veloc_Processador': [2.73],
    'Capac_Bateria': [4500],
    'Carreg_Rápido_Disp': [1],
    'Carreg_Rápido': [25],
    'Capacidade_Ram': [8],
    'Memória_Interna': [128],
    'Tamanho_Tela': [6.7],
    'Taxa_Atualização': [120],
    'Qtd_Câm_Tras': [4],
    'Qtd_Câm_Front': [1],
    'Sistema_Operacional': ['android'],
    'Câm_Tras_Principal': [64],
    'Câm_Front_Principal': [10],
    'Memória_Esten_Disp': [1],
    'Expansível_Até': [1024]
})

# Pré-processamento dos novos dados
new_data_processed = preprocess_new_data(new_data)

# Previsão com o modelo SVM
new_prediction = svm.predict(new_data_processed)

print('Predição de "Avaliação" para novos dados:', new_prediction[0])
