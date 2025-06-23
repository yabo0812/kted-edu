# app/server.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

from langserve import add_routes

# FastAPI 서버를 설정
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# 라우팅 설정
add_routes(
    app,
    ChatOpenAI(model="gpt-4.1-mini"),
    path="/openai",  # OpenAI 모델에 대한 경로
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
