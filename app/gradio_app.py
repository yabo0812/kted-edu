import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 환경변수 로드
load_dotenv()

######################
#  RAG 체인 구성
######################

# OpenAI 임베딩 모델 생성
embeddings_openai = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# 저장된 벡터 저장소를 가져오기
chroma_db = Chroma(
    collection_name="labor_law",
    embedding_function=embeddings_openai,
    persist_directory="./chroma_db",
)

print("Chroma DB loaded")
print(chroma_db._collection.count())  # 벡터 저장소에 있는 문서 수 출력

# 검색기 초기화
retriever = chroma_db.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k': 5,                  # 검색할 문서의 수
        'fetch_k': 10,           # mmr 알고리즘에 전달할 문서의 수 (fetch_k > k)
        'lambda_mult': 0.3,      # 다양성을 고려하는 정도
    },
)

# 메시지 플레이스홀더가 있는 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", """주어진 컨텍스트를 기반으로 질문에 답변하시오.

[지침]
- 컨텍스트에 있는 정보만을 사용하여 답변할 것
- 외부 지식이나 정보를 사용하지 말 것
- 컨텍스트에서 답을 찾을 수 없는 경우 "주어진 정보만으로는 답변하기 어렵습니다."라고 응답할 것
- 불확실한 경우 명확히 그 불확실성을 표현할 것
- 답변은 논리적이고 구조화된 형태로 제공할 것
- 답변은 한국어를 사용할 것"""),
    MessagesPlaceholder("chat_history"),
    ("system", """[컨텍스트]
{context}

이전 대화 내용을 참고하여 질문에 대해서 친절하게 답변합니다.

[답변 형식]
1. 핵심 답변: (질문에 대한 직접적인 답변)
2. 근거: (컨텍스트에서 발견된 관련 정보)
3. 추가 설명: (필요한 경우 부연 설명 제공)"""),
    ("human", "{question}")
])

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
rag_chain = prompt | llm | StrOutputParser()

# 사용자 메시지를 처리하고 AI 응답을 생성하는 함수
def answer_invoke(message, history):
    # 대화 이력을 LangChain 메시지 형식으로 변환
    history_messages = []
    for msg in history:
        if msg['role'] == "user":
            history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_messages.append(AIMessage(content=msg['content']))
    
    # RAG 체인 실행
    response = rag_chain.invoke({
        "chat_history": history_messages,
        "context": format_docs(retriever.invoke(message)), 
        "question": message
    })
    
    return response

# Gradio ChatInterface 객체 생성
demo = gr.ChatInterface(
    fn=answer_invoke,                    # 메시지 처리 함수
    title="근로기준법 Q&A 챗봇",              # 채팅 인터페이스의 제목
    description="근로기준법 관련 질문에 답변하는 AI 챗봇입니다.",
    examples=[
        "근로계약서에는 어떤 내용이 포함되어야 하나요?",
        "연차휴가는 어떻게 계산하나요?",
        "최저임금은 어떻게 정해지나요?",
        "해고 절차는 어떻게 되나요?"
    ],
    type="messages"
)

# Gradio 인터페이스 실행
if __name__ == "__main__":
    demo.launch()