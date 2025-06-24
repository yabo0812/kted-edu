from fastapi import FastAPI
from dotenv import load_dotenv
from app.rag import rag_chain
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 환경변수 로드
load_dotenv()

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

add_routes(
    app,
    rag_chain,
    path="/rag",  # RAG 체인에 대한 경로
)

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
