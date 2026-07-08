import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:/Users/nosso/Desktop/Analise de Dados/Inteligência Artificial/hospital_ai_project/data/hospital_data.csv")

sns.countplot(x="prioridade", data=data)
plt.title("Distribuição de Prioridade")
plt.show()

sns.histplot(data["tempo_espera"], bins=30)
plt.title("Tempo de Espera")
plt.show()