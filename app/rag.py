# app/rag.py
from dotenv import load_dotenv

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 환경변수 로드
load_dotenv()


######################
#  RAG 체인 구성
######################

# OpenAI 임베딩 모델 생성
embeddings_openai = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 사용할 모델 이름
)

# 저장된 벡터 저장소를 가져오기
chroma_db = Chroma(
    collection_name="labor_law",
    embedding_function=embeddings_openai,
    persist_directory="./chroma_db",
)

print("Chroma DB loaded")
print(chroma_db._collection.count())  # 벡터 저장소에 있는 문서 수 출력

# 검색기 초기화å
retriever = chroma_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,  # 검색할 문서의 수
        "fetch_k": 10,  # mmr 알고리즘에 전달할 문서의 수 (fetch_k > k)
        "lambda_mult": 0.3,  # 다양성을 고려하는 정도 (1은 최소 다양성, 0은 최대 다양성을 의미. 기본값은 0.5)
    },
)

# Prompt 템플릿 생성
template = """주어진 컨텍스트를 기반으로 질문에 답변하시오.

[지침]
- 컨텍스트에 있는 정보만을 사용하여 답변할 것
- 외부 지식이나 정보를 사용하지 말 것
- 컨텍스트에서 답을 찾을 수 없는 경우 "주어진 정보만으로는 답변하기 어렵습니다."라고 응답할 것
- 불확실한 경우 명확히 그 불확실성을 표현할 것
- 답변은 논리적이고 구조화된 형태로 제공할 것
- 답변은 한국어를 사용할 것

[컨텍스트]
{context}

[질문]
{question}

[답변 형식]
1. 핵심 답변: (질문에 대한 직접적인 답변)
2. 근거: (컨텍스트에서 발견된 관련 정보)
3. 추가 설명: (필요한 경우 부연 설명 제공)

[답변]
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM 설정
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    top_p=0.9,
)


# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join([f"{doc.page_content}" for doc in docs])


# RAG 체인 생성

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
