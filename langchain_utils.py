import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import login
from fastapi import HTTPException

# 환경 변수 로드
load_dotenv()

class LangChainHelper:
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.chat_history = []

        # Hugging Face 로그인
        huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        login(huggingface_api_key)

    def process_pdf(self, file_path: str):
        # PDF 파일 처리
        try:
            loader = PyPDFLoader(file_path)
            data = loader.load()
            text = "".join([doc.page_content for doc in data])
            print("PDF 파일이 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"Error loading PDF file: {e}")
            raise HTTPException(status_code=500, detail="PDF 파일을 로드하는 중 오류가 발생했습니다.")

        # 텍스트 분할 및 벡터 저장소 생성
        try:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            self.vector_store = Chroma.from_texts(chunks, self.embeddings)
            print("벡터 저장소가 성공적으로 생성되었습니다.")
        except Exception as e:
            print(f"Error processing text and creating vector store: {e}")
            raise HTTPException(status_code=500, detail="텍스트를 처리하고 벡터 저장소를 생성하는 중 오류가 발생했습니다.")

    async def get_answer(self, question: str) -> str:
        if self.vector_store is None:
            return "먼저 PDF 파일을 업로드하세요."

        try:
            # 유사한 문서 검색, 결과 수를 2개로 제한
            docs = self.vector_store.similarity_search(question, k=2)
            
            # 검색된 결과에서 텍스트 길이 제한 및 요약
            context_text = " ".join([doc.page_content[:1000] for doc in docs])

            # 컨텍스트 길이 제한 (예: 1500자로 제한)
            max_context_length = 1500
            if len(context_text) > max_context_length:
                context_text = context_text[:max_context_length]
                print("컨텍스트가 너무 길어 잘라서 사용합니다.")

            # LLM 설정 및 응답 생성
            llm = ChatOpenAI(api_key=self.openai_api_key, model="gpt-3.5-turbo", temperature=0.1)
            chain = ConversationalRetrievalChain.from_llm(llm, self.vector_store.as_retriever())

            # ChatGPT에 질문과 요약된 컨텍스트 전달
            inputs = {
                "question": question,
                "chat_history": self.chat_history,
                "context": context_text
            }
            result = chain.run(inputs)
            
            # 대화 기록에 추가
            self.chat_history.append((question, result))
            return result
        except Exception as e:
            print(f"Error getting answer: {e}")
            return "질문에 대한 답변을 생성하는 중 오류가 발생했습니다."
