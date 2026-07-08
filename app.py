from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI()

# =========================
# 1. Estrutura do input (profissional)
# =========================
class Paciente(BaseModel):
    idade: float
    sexo: float
    frequencia_cardiaca: float
    pressao_sistolica: float
    temperatura: float
    historico_doenca: float
    ocupacao_hospital: float
    tempo_chegada: float

# =========================
# 2. Modelo (IGUAL ao treinamento)
# =========================
class RedeHospital(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 🔥 IMPORTANTE (igual ao treino)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

# =========================
# 3. Carregar modelo
# =========================
model = RedeHospital(8)
model.load_state_dict(torch.load("C:/Users/nosso/Desktop/Analise de Dados/Inteligência Artificial/hospital_ai_project/models/modelo_hospital.pth"))
model.eval()

# =========================
# 4. Rota inicial
# =========================
@app.get("/")
def home():
    return {"mensagem": "API Hospital IA rodando"}

# =========================
# 5. Previsão
# =========================
@app.post("/prever")
def prever(paciente: Paciente):
    
    dados = [
        paciente.idade,
        paciente.sexo,
        paciente.frequencia_cardiaca,
        paciente.pressao_sistolica,
        paciente.temperatura,
        paciente.historico_doenca,
        paciente.ocupacao_hospital,
        paciente.tempo_chegada
    ]

    entrada = torch.tensor([dados], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(entrada)
        _, pred = torch.max(output, 1)
    
    return {"prioridade_prevista": int(pred.item())}

