#################################
# 1. 기본 설정 및 라이브러리 임포트
#################################

# SQLite3 관련 설정
import sys
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 필수 라이브러리 임포트
import streamlit as st  # 웹 인터페이스 생성용
import openai  # OpenAI API 사용
import os
import time
import json
import logging

# LangChain 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI  # 경로 수정
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 경로 수정
from langchain_community.vectorstores import FAISS  # 경로 수정
from langchain_core.prompts import ChatPromptTemplate  # 경로 수정
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document  # 경로 수정
from langchain_core.output_parsers import StrOutputParser  # 경로 수정
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # 경로 수정

# .env 파일 사용을 위한 import 추가
from dotenv import load_dotenv
load_dotenv()

#################################
# 2. 초기 설정
#################################

# API 키는 환경변수에서 가져오기
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 로깅 설정
logging.basicConfig(
    filename='chatbot.log',  # 로그 파일명을 더 명확하게 변경
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a',
    encoding='utf-8'  # 한글 로깅을 위한 인코딩 설정
)

#################################
# 3. 데이터 준비
#################################

# ChatGPT 모델 설정 (온도값이 낮을수록 더 사실적인 답변)
try:
    # OpenAI API 키 확인
    if not openai.api_key:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
    # 모델 설정 (GPT-4 Mini 모델명으로 수정)
    chat_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1)
except Exception as e:
    st.error(f"API 설정 오류: {str(e)}")
    st.stop()

try:
    # 위키피디아에서 데이터 로드
    wiki_loader = WebBaseLoader("https://en.wikipedia.org/wiki/Elon_Musk")
    wiki_data = wiki_loader.load()

    # 데이터를 작은 조각으로 분할 (더 효율적인 처리를 위해)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,  # 각 텍스트 조각의 크기
        chunk_overlap=2000  # 조각 간 겹치는 부분 (문맥 유지를 위해)
    )
    split_texts = text_splitter.split_documents(wiki_data)

except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {str(e)}")
    st.stop()

#################################
# 4. 벡터 데이터베이스 설정
#################################

# FAISS 인덱스 저장 경로 설정
FAISS_INDEX_PATH = "faiss_index"

# 벡터 데이터베이스 생성 또는 로드
try:
    # 텍스트를 벡터로 변환하는 모델 설정
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # 벡터 데이터베이스 생성
    vector_db = FAISS.from_documents(
        documents=split_texts,
        embedding=embeddings_model
    )
    vector_db.save_local("faiss_index")

    # 다중 질의 검색기 설정 (더 정확한 답변을 위해 질문을 여러 방식으로 해석)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(
            search_type="mmr",  # MMR: 다양성과 관련성을 모두 고려하는 검색 방식
            search_kwargs={'k': 5, 'fetch_k': 50}
        ),
        llm=chat_model
    )

except Exception as e:
    st.error(f"벡터 데이터베이스 처리 중 오류 발생: {str(e)}")
    st.stop()

#################################
# 5. 챗봇 프롬프트 설정
#################################

# 챗봇의 응답 방식을 정의하는 템플릿
chat_template = """From now on, you are an expert who finds and provides information that fits the user's question in the provided context.

1. When you want information that is not in the context, answer that you do not have the information you want.
2. Never answer with information that is not in the context.
3. If you do not know the content, say that you do not know.
4. Make the explanation as detailed and long as possible.
5. Make sure to answer in Korean!!

context: {context}
user input: {input}
"""

prompt = ChatPromptTemplate.from_template(chat_template)

# 검색과 응답 생성을 위한 처리 과정 설정
retrieval_chain = RunnableParallel({"context": vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50}), "input": RunnablePassthrough()})
response_chain = retrieval_chain | prompt | chat_model | StrOutputParser()

#################################
# 6. 웹 인터페이스 구성
#################################

# 웹페이지 제목 설정
st.title("일론 머스크 AI 챗봇")
st.markdown("_위키피디아의 일론 머스크 정보를 기반으로 답변하는 챗봇입니다_")

# 세션 상태 초기화
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini-2024-07-18"

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 상수 정의를 파일 상단으로 이동
MAX_MESSAGES = 4
MAX_CHUNK_SIZE = 10000
CHUNK_OVERLAP = 2000

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요"):
    # 메시지 개수 제한
    if len(st.session_state.messages) >= MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-2:]

    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답 생성 및 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # AI 응답 생성
            result = response_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.messages
            })

            # 타이핑 효과로 응답 표시
            for chunk in result.split(" "):
                full_response += chunk + " "
                time.sleep(0.1)  # 타이핑 속도를 조금 더 빠르게 조정
                message_placeholder.markdown(full_response + " ")
            message_placeholder.markdown(full_response)

        except openai.error.InvalidRequestError as e:
            st.error("OpenAI API 요청 오류: 입력이 너무 깁니다.")
            logging.error(f"OpenAI API 오류: {str(e)}")
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            logging.error(f"일반 오류: {str(e)}")

    # AI 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 디버깅을 위한 대화 기록 출력
logging.info("=== 대화 기록 ===")
logging.info(st.session_state.messages)