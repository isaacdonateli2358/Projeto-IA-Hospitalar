@echo off
echo Iniciando API Hospital IA...
cd /d "C:\Users\nosso\Desktop\Analise de Dados\Inteligência Artificial\hospital_ai_project\api"
uvicorn app:app --reload
pause