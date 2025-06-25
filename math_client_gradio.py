import asyncio
import threading
import gradio as gr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

class SimpleMCPClient:
    def __init__(self):
        self.agent = None
        self.is_connected = False
        self.stdio_context = None
        self.session_context = None
        self.session = None
        
    async def connect_to_server(self, server_path: str):
        """MCP 서버에 연결하고 에이전트를 초기화"""
        try:
            # 기존 연결이 있으면 해제
            await self.disconnect()
            
            server_params = StdioServerParameters(
                command="python",
                args=[server_path],
                env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            )
            
            # MCP 클라이언트 연결
            self.stdio_context = stdio_client(server_params)
            self.read, self.write = await self.stdio_context.__aenter__()
            
            self.session_context = ClientSession(self.read, self.write)
            self.session = await self.session_context.__aenter__()
            
            # 세션 초기화
            await self.session.initialize()
            
            # 도구 로드
            tools = await load_mcp_tools(self.session)
            
            # 에이전트 생성 (올바른 모델명 사용)
            self.agent = create_react_agent("openai:gpt-4.1-mini", tools)
            self.is_connected = True
            
            tool_names = [tool.name for tool in tools] if tools else []
            return f"✅ MCP 서버에 성공적으로 연결되었습니다.\n사용 가능한 도구 ({len(tools)}개): {', '.join(tool_names)}"
            
        except Exception as e:
            self.is_connected = False
            return f"❌ 연결 실패: {str(e)}\n서버 파일 경로와 권한을 확인해주세요."
    
    async def process_message(self, message: str):
        """메시지를 처리하고 응답 반환"""
        if not self.is_connected or not self.agent:
            return "❌ 먼저 MCP 서버에 연결해주세요."
        
        try:
            # 메시지 형식을 올바르게 구성
            messages = [{"role": "user", "content": message}]
            response = await self.agent.ainvoke({"messages": messages})
            
            # 응답에서 마지막 메시지 추출
            if response and "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    return last_message['content']
                else:
                    return str(last_message)
            else:
                return "응답을 받지 못했습니다. 다시 시도해주세요."
                
        except Exception as e:
            return f"❌ 오류가 발생했습니다: {str(e)}"
    
    async def disconnect(self):
        """서버 연결 해제"""
        self.is_connected = False
        try:
            if self.session_context:
                await self.session_context.__aexit__(None, None, None)
                self.session_context = None
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
                self.stdio_context = None
        except Exception:
            pass

# 전역 클라이언트 인스턴스와 이벤트 루프
client = SimpleMCPClient()
loop = None

def get_or_create_loop():
    """이벤트 루프를 가져오거나 생성"""
    global loop
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        # 새 스레드에서 루프 실행
        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
    return loop

def connect_server(server_path):
    """서버 연결 함수 (동기)"""
    if not server_path.strip():
        return "❌ 서버 경로를 입력해주세요."
    
    try:
        event_loop = get_or_create_loop()
        future = asyncio.run_coroutine_threadsafe(
            client.connect_to_server(server_path), 
            event_loop
        )
        result = future.result(timeout=30)  # 30초 타임아웃
        return result
    except asyncio.TimeoutError:
        return "❌ 연결 시간 초과. 서버가 실행 중인지 확인해주세요."
    except Exception as e:
        return f"❌ 연결 오류: {str(e)}"

def chat_response(message, history):
    """채팅 응답 함수 (동기)"""
    if not message.strip():
        return history, ""
    
    try:
        event_loop = get_or_create_loop()
        future = asyncio.run_coroutine_threadsafe(
            client.process_message(message), 
            event_loop
        )
        response = future.result(timeout=60)  # 60초 타임아웃
        
        # Gradio chatbot의 messages 형식에 맞게 구성
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        return new_history, ""
        
    except asyncio.TimeoutError:
        error_msg = "⏱️ 응답 시간이 초과되었습니다. 다시 시도해주세요."
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        return new_history, ""
    except Exception as e:
        error_msg = f"❌ 오류: {str(e)}"
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        return new_history, ""

def clear_chat():
    """채팅 내역 지우기"""
    return []

def disconnect_server():
    """서버 연결 해제"""
    try:
        event_loop = get_or_create_loop()
        future = asyncio.run_coroutine_threadsafe(client.disconnect(), event_loop)
        future.result(timeout=10)
        return "🔌 서버 연결이 해제되었습니다."
    except Exception as e:
        return f"연결 해제 중 오류: {str(e)}"

# Gradio 인터페이스 구성
with gr.Blocks(title="MCP Chat Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 MCP Chat Assistant")
    gr.Markdown("MCP 서버에 연결하여 도구를 사용할 수 있는 AI 어시스턴트입니다.")
    
    with gr.Row():
        with gr.Column(scale=3):
            server_path = gr.Textbox(
                label="MCP 서버 경로",
                placeholder="/path/to/your/server.py",
                value="math_server.py"
            )
        with gr.Column(scale=1):
            connect_btn = gr.Button("연결", variant="primary")
            disconnect_btn = gr.Button("연결 해제", variant="secondary")
    
    status_display = gr.Textbox(
        label="연결 상태", 
        interactive=False,
        value="🔴 서버에 연결되지 않음"
    )
    
    chatbot = gr.Chatbot(
        label="채팅",
        height=400,
        show_copy_button=True,
        type="messages",  # messages 타입 사용
        avatar_images=("👤", "🤖")
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            msg_input = gr.Textbox(
                label="메시지 입력",
                placeholder="질문을 입력하세요... (예: 3 + 5 곱하기 12는?)",
                lines=2
            )
        with gr.Column(scale=1):
            send_btn = gr.Button("전송", variant="primary")
            clear_btn = gr.Button("대화 지우기")
    
    # 예시 질문 버튼들
    with gr.Row():
        example_btn1 = gr.Button("예시: 3 + 5 × 12 계산해줘", size="sm")
        example_btn2 = gr.Button("예시: 22 더하기 8은 얼마죠?", size="sm")
        example_btn3 = gr.Button("예시: 7과 6을 곱한 값은?", size="sm")

    # 이벤트 핸들러 연결
    connect_btn.click(
        connect_server,
        inputs=[server_path],
        outputs=[status_display]
    )
    
    disconnect_btn.click(
        disconnect_server,
        outputs=[status_display]
    )
    
    msg_input.submit(
        chat_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    
    send_btn.click(
        chat_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot]
    )
    
    # 예시 버튼 이벤트
    example_btn1.click(
        lambda: "3 + 5 × 12 계산해줘",
        outputs=[msg_input]
    )
    example_btn2.click(
        lambda: "22 더하기 8은 얼마죠?",
        outputs=[msg_input]
    )
    example_btn3.click(
        lambda: "7과 6을 곱한 값은?",
        outputs=[msg_input]
    )

if __name__ == "__main__":
    print("🚀 MCP Chat Assistant 시작 중...")
    print("📋 필요한 환경 변수: OPENAI_API_KEY")
    demo.launch(
        debug=True,
        share=False,
        inbrowser=True,
        server_port=7860
    )