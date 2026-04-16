# IA para Otimização de Atendimento Hospitalar

Projeto de Inteligência Artificial desenvolvido para melhorar a eficiência no atendimento hospitalar por meio da classificação automática de prioridade de pacientes.

---

## Objetivo

Desenvolver um modelo de rede neural capaz de:

* Classificar a prioridade de atendimento (normal, urgente, emergência)
* Auxiliar na triagem de pacientes
* Reduzir tempo de espera
* Apoiar decisões operacionais em hospitais

---

## Tecnologias utilizadas

* Python
* PyTorch
* FastAPI
* Pandas
* NumPy
* Scikit-learn

---

## Modelo de IA

Foi desenvolvido um modelo de rede neural multicamadas (MLP) com:

* Camadas densas (Linear)
* Função de ativação ReLU
* Dropout para evitar overfitting
* Treinamento supervisionado

### Resultados

* Acurácia: 81%
* Bom desempenho na identificação de casos críticos (emergência)
* Uso de balanceamento de classes para melhorar o modelo

---

## Variáveis de entrada

O modelo utiliza os seguintes dados do paciente:

* Idade
* Sexo
* Frequência cardíaca
* Pressão arterial sistólica
* Temperatura
* Histórico de doenças
* Ocupação do hospital
* Tempo de chegada

---

## Saída

O modelo retorna a classificação da prioridade:

* 0 → Normal
* 1 → Urgente
* 2 → Emergência

---

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/hospital-ai-project.git
cd hospital-ai-project/api
```

---

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

---

### 3. Executar a API

```bash
uvicorn app:app --reload
```

---

### 4. Acessar a documentação

Abra no navegador:

http://127.0.0.1:8000/docs

---

## Exemplo de uso

### Entrada

```json
{
  "idade": 70,
  "sexo": 1,
  "frequencia_cardiaca": 140,
  "pressao_sistolica": 160,
  "temperatura": 39,
  "historico_doenca": 1,
  "ocupacao_hospital": 0.9,
  "tempo_chegada": 10
}
```

### Saída

```json
{
  "prioridade_prevista": 2
}
```

---

## Diferenciais do projeto

* Aplicação prática de Inteligência Artificial na área da saúde
* API funcional com previsão em tempo real
* Estrutura pronta para integração com sistemas hospitalares
* Projeto voltado para resolução de problemas reais

---

## Próximos passos

* Deploy em nuvem (Render ou similar)
* Integração com dashboards (Power BI)
* Utilização de dados reais hospitalares
* Expansão do modelo para prever tempo de espera

---

## Autor

Isaac Donateli
Analista de Dados | BI | Inteligência Artificial

---

