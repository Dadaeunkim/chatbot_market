import gradio as gr
from langchain_openai import ChatOpenAI # openAI API 사용
from langchain.memory import ConversationBufferMemory # 메모리 저장
from langchain.chains import ConversationChain # 데이터와 메모리 연결
from langchain.schema import AIMessage, HumanMessage, SystemMessage #채팅 메시지 구분
from langchain.document_loaders import TextLoader, PyPDFLoader
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#LLM과 메모리 초기화
llm=ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo')
memory=ConversationBufferMemory(return_messages=True)

# 대화 내용을 기억하기 위한 준비
conversation=ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

# 상담봇 - 채팅 및 답변 
def counseling_bot_chat(message, chat_history): # message:  #3 입력 내용, chat_history: #2 채팅 내역
    if message == "": 
        return "", chat_history 
        
    else:
        result_message="" 
        # 처음 대화인 경우 api 호출하여 상담봇 역할 부여
        if len(chat_history)<=1: 
            # system 프롬프트를 memory에 수동으로 삽입
            memory.chat_memory.add_user_message("시스템 초기화")
            memory.chat_memory.add_ai_message("당신은 헤이마트의 상담원입니다. 마트 상품과 관련되지 않은 질문에는 정중히 거절하세요.")
            result_message=conversation.predict(input=message)
    
        # SystemMessage가 추가 되어 있는 상태이므로 곧바로 cconversation.predict 실행   
        else:
            result_message=conversation.predict(input=message) # messages 아님
            
        chat_history.append([message, result_message])
        return " ", chat_history

# 상담봇 - 되돌리기 
def counseling_bot_undo(chat_history): 
    if len(chat_history) > 1: 
        chat_history.pop() 
    return chat_history 

# 상담봇 - 초기화 
def counseling_bot_reset(chat_history): 
    chat_history=[[None,"안녕하세요, 헤이마트입니다. 상담을 도와드리겠습니다."]]
    return chat_history 

# 번역봇
def translate_bot(output_conditions, output_language, input_text):
    if input_text =="": 
        return "" 
    else:
        if output_conditions=="": 
            completion=client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system", 
                           "content":"""당신은 번역가입니다.
                            입력한 언어를 다른 설명 없이 곧바로 {0}로 번역해서 알려 주세요.
                            번역이 불가능한 언어라면 번역이 불가능하다고 말한 후 그 이유를 설명해 주세요.""".format(output_language)
                          },
                          {"role":"user", 
                           "content":input_text
                          }
                         ]
            )
        else: 
            output_conditions="번역할 때의 조건은 다음과 같습니다." + output_conditions 
            
            completion=client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system", 
                           "content":"""당신은 번역가입니다.
                            입력한 언어를 다른 설명 없이 곧바로 {0}로 번역해서 알려 주세요.
                            번역이 불가능한 언어라면 번역이 불가능하다고 말한 후 그 이유를 설명해 주세요.{1}""".format(output_language, output_conditions)
                          },
                          {"role":"user", 
                           "content":input_text
                          }
                         ]
            )
        return completion.choices[0].message.content

# 번역봇 - Text 업로드
def translate_bot_Text_upload(files): 
    loader=TextLoader(files, encoding='utf-8')
    document=loader.load() 
    return document[0].page_content

# 번역봇 - PDF 업로드
def translate_bot_PDF_upload(files): 
    loader=PyPDFLoader(files) 
    pages=loader.load_and_split() 
    return pages[0].page_content

# 레이아웃
with gr.Blocks(theme=gr.themes.Default()) as app:       # 블록 테마 기본으로 지정, 사용자 지정하려면 허깅페이스의 주소 입력
    with gr.Tab("상담봇"):
        # 레이아웃 구성: 세로 1 블록, 가로 4블록, 3,4번째 블록에 세로 2블록
        #기본값으로 세로 블록 1개 쌓음, gr.Columns() 생략 가능
        #1 - 타이틀 및 설명 
        gr.Markdown(
            value=""" 
            # <center>상담봇</center> 
            <center>헤이마트 상담봇입니다. 마트에서 판매하는 상품과 관련된 질문에 답변드립니다.</center>
            """    # value 값에 마크다운 문법 작성
        )               
        #2 - 채팅 화면 
        cb_chatbot=gr.Chatbot(
            value=[[None, "안녕하세요, 헤이마트입니다. 상담을 도와드리겠습니다."]], 
            show_label=False 
        )              # value 값 형태: [[유저 인풋0, 챗봇 인풋0], [유저 인풋1, 챗봇 인풋1], ...]
        with gr.Row(): # 가로 블록 1
            #3 - 입력 창
            cb_user_input=gr.Text(
                lines=1, 
                placeholder="질문을 입력해주세요", 
                container=False, # 바깥 테두리와 텍스트박스 label 제거
                scale=9  # row 안의 컴포넌트 간의 비중
            ) 
            #4 -보내기 버튼
            cb_send_btn=gr.Button(
                value="보내기", 
                scale=1,    # text와 buttom 비중 9:1
                variant="primary",     # 버튼스타일
               # icon=""
            ) 
        with gr.Row(): # 가로 블록 2
            #5 -되돌리기 버튼
            gr.Button(value="↩ 되돌리기").click(fn=counseling_bot_undo, inputs=cb_chatbot, outputs=cb_chatbot) # fn:함수, inputs: fn의 매개변수값, outputs: fn의 return값
            #6 - 초기화 버튼 
            gr.Button(value="🔄 초기화").click(fn=counseling_bot_reset, inputs=cb_chatbot, outputs=cb_chatbot)
            # 보내기 1 
        cb_send_btn.click(fn=counseling_bot_chat, inputs=[cb_user_input, cb_chatbot], outputs=[cb_user_input, cb_chatbot]) # 보내기 버튼을 클릭한 경우 발생할 이벤트
            # 보내기 2 
        cb_user_input.submit(fn=counseling_bot_chat, inputs=[cb_user_input, cb_chatbot], outputs=[cb_user_input, cb_chatbot]) # 입력창에 글 쓴후 엔터를 눌렀을때 발생할 이벤트 
        
    with gr.Tab("번역봇"): 
        #1 타이틀 및 설명
        gr.Markdown(
            value=""" 
            # <center>번역봇<center/>
            <center>다국어 번역 봇입니다.<center/>
            """
        )
        with gr.Row():
            #2 번역 조건
            tb_output_condition=gr.Text(
                label="번역 조건", 
                placeholder="예시: 자연스럽게",
                lines=1, 
                max_lines=3
            )
            #3 출력 언어 
            tb_output_language=gr.Dropdown(
                label="출력 언어", 
                choices=["한국어","영어","일본어","중국어"], 
                value="한국어", # 초기값
                allow_custom_value=True, #언어 직접 설정 가능하도록
                interactive=True
            )
        with gr.Row():
            #7
            tb_TXTupload=gr.UploadButton(label="📄 TXT 업로드")
            #8
            tb_PDFupload=gr.UploadButton(label="📤 PDF 업로드")
        #4 번역하기 버튼
        tb_submit=gr.Button(
            value="번역하기", 
            variant="primary",
        )

        with gr.Row(): 
            #5 입력 텍스트 필드
            tb_input_text=gr.Text(
                placeholder="번역할 내용을 적어 주세요.", 
                lines=10, 
                max_lines=20, 
                show_copy_button=True, 
                label="입력"
            )
            #6 출력 텍스트 필드 
            tb_output_text=gr.Text(
                lines=10,
                max_lines=20,
                show_copy_button=True, 
                label="", 
                interactive=False
            )
            #번역봇 내용 보내기
            tb_submit.click(
                fn=translate_bot, 
                inputs=[tb_output_condition, tb_output_language, tb_input_text], 
                outputs=tb_output_text
            )
            #Text 파일 업로드 
            tb_TXTupload.upload(
                 fn=translate_bot_Text_upload,
                 inputs=tb_TXTupload,
                 outputs=tb_input_text
             )
            # PDF 파일 업로드 
            tb_PDFupload.upload(
                fn=translate_bot_PDF_upload, 
                inputs=tb_PDFupload, 
                outputs=tb_input_text
            )
            
    with gr.Tab("소설봇"): 
        pass
app.launch()