import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000

data = pd.DataFrame({
    "idade": np.random.randint(0, 100, n),
    "sexo": np.random.choice([0, 1], n),  # 0=fem, 1=masc
    "frequencia_cardiaca": np.random.randint(60, 180, n),
    "pressao_sistolica": np.random.randint(90, 180, n),
    "temperatura": np.random.uniform(35, 40, n),
    "historico_doenca": np.random.choice([0, 1], n),
    "ocupacao_hospital": np.random.uniform(0.3, 1.0, n),
    "tempo_chegada": np.random.randint(0, 24, n)
})

# Criando regra de prioridade (simulada)
def definir_prioridade(row):
    if row["frequencia_cardiaca"] > 130 or row["temperatura"] > 38.5:
        return 2  # emergência
    elif row["pressao_sistolica"] > 150:
        return 1  # urgente
    else:
        return 0  # normal

data["prioridade"] = data.apply(definir_prioridade, axis=1)

# Tempo de espera (target regressão)
data["tempo_espera"] = (
    60 * data["ocupacao_hospital"] +
    np.random.normal(0, 10, n)
).clip(5, 180)

# Tipo de atendimento
data["tipo_atendimento"] = np.random.choice([0, 1, 2, 3], n)

data.to_csv("C:/Users/nosso/Desktop/Analise de Dados/Inteligência Artificial/hospital_ai_project/data/hospital_data.csv", index=False),

data.head()
