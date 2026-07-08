import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# =========================
# 1. Carregar dados
# =========================
data = pd.read_csv("C:/Users/nosso/Desktop/Analise de Dados/Inteligência Artificial/hospital_ai_project/data/hospital_data.csv")  # caminho relativo

# =========================
# 2. Separar features e target
# =========================
X = data.drop(columns=["prioridade", "tempo_espera", "tipo_atendimento"])
y = data["prioridade"]

# =========================
# 3. Normalização
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 4. Dividir dados
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Converter para tensor
# =========================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# =========================
# 6. Calcular pesos (DESBALANCEAMENTO)
# =========================
classes = np.unique(y_train.numpy())
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train.numpy()
)
weights = torch.tensor(weights, dtype=torch.float32)

# =========================
# 7. Criar modelo
# =========================
class RedeHospital(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        return self.model(x)

model = RedeHospital(X_train.shape[1])

# =========================
# 8. Loss e Otimizador
# =========================
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 9. Treinamento
# =========================
epochs = 100

for epoch in range(epochs):
    model.train()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# =========================
# 10. Avaliação
# =========================
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"\nAcurácia: {accuracy:.2f}")

# =========================
# 11. Relatório completo
# =========================
print("\nRelatório de Classificação:")
print(classification_report(y_test.numpy(), predicted.numpy()))

# =========================
# 12. Salvar modelo
# =========================
torch.save(model.state_dict(), "C:/Users/nosso/Desktop/Analise de Dados/Inteligência Artificial/hospital_ai_project/models/modelo_hospital.pth")

print("\n✅ Modelo salvo com sucesso em /models/")