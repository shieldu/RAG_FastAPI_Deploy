# Langchain_PDF_ChatBot 📄🤖

## 설명
이 프로젝트는 허깅페이스의 임베딩 모델(sentence-transformers/all-MiniLM-L6-v2)을 사용하여 PDF 파일을 임베딩합니다.
이후, 임베딩된 데이터를 벡터 DB(ChromaDB)에 저장합니다. 사용자가 질문을 입력하면, 시스템은 관련 데이터를 벡터 DB에서 검색합니다.
검색된 데이터를 바탕으로 ChatGPT API를 통해 답변을 생성하는 시스템입니다.

<br>

## 설치 및 실행 방법

### 요구 사항
프로그램을 실행하기 위해 필요한 라이브러리들이 `requirements.txt` 파일에 명시되어 있습니다. 다음 명령어로 필요한 라이브러리들을 설치하세요:

```bash
pip install -r requirements.txt
```

`requirements.txt` 파일에는 다음 라이브러리들이 포함되어 있습니다:
- flask: 웹 프레임워크
- langchain: 자연어 처리 라이브러리
- chromadb: 벡터 DB
- sentence-transformers: HuggingFace 임베딩 모델
- openai: OpenAI API를 위한 패키지
- pypdf: PDF 처리용 패키지
- python-dotenv: 환경 변수 로드용

<br>

### 실행 방법
1. OpenAI API 키를 `.env` 파일에 저장하세요:
```
OPENAI_API_KEY=your_openai_api_key
```

2. `app.py` 파일을 실행하여 웹 애플리케이션을 시작하세요:
```bash
python app.py
```

3. 웹 브라우저에서 `http://localhost:5000`으로 이동하여 PDF 파일을 업로드하고 질문을 입력하세요.

<br>

### PDF 업로드 및 질문
- PDF 파일을 업로드한 후, 해당 문서의 내용을 바탕으로 질문을 입력하면 챗봇이 답변을 제공합니다.

<br>

## 주요 함수 설명

#### PDF 파일 업로드 및 처리 기능
PDF 파일을 서버에 저장한 후, PyPDFLoader를 통해 텍스트로 변환합니다. 텍스트는 1000자 단위로 분할되며, 중복된 200자를 포함하여 문맥을 유지합니다. 생성된 텍스트 조각은 Hugging Face 임베딩 모델로 벡터화된 후 Chroma DB에 저장됩니다. Chroma DB는 유사한 문서를 빠르게 검색하여 ChatGPT API로 답변을 생성할 수 있도록 지원합니다.

```python
def process_pdf(self, pdf_file):
    file_path = os.path.join('uploads', pdf_file.filename)
    pdf_file.save(file_path)

    loader = PyPDFLoader(file_path)
    data = loader.load()

    text = "".join([doc.page_content for doc in data])
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    self.vector_store = Chroma.from_texts(chunks, self.embeddings)
```
<br>

#### 질문에 대한 답변 생성
사용자가 입력한 질문을 벡터 DB에서 검색하여 관련 문서를 찾습니다. GPT-3.5-turbo 모델은 검색된 문서를 바탕으로 답변을 생성합니다. 대화 기록인 `chat_history`를 사용해 문맥을 유지하며 일관성 있는 답변을 제공합니다.

```python
def get_answer(self, question):
    if self.vector_store is None:
        return "먼저 PDF 파일을 업로드하세요."

    docs = self.vector_store.similarity_search(question)

    llm = ChatOpenAI(api_key=self.openai_api_key, model="gpt-3.5-turbo", temperature=0.1)
    chain = ConversationalRetrievalChain.from_llm(llm, self.vector_store.as_retriever())

    inputs = {"question": question, "chat_history": self.chat_history}
    result = chain.run(inputs)

    self.chat_history.append((question, result))
    return result
```
<br>

#### Flask API 호출
이 시스템은 PDF 파일 업로드와 질문에 대한 답변을 제공하는 두 가지 API를 통해 사용자와 상호작용합니다. PDF 업로드 API는 사용자가 업로드한 파일을 처리하고, 질문 API는 GPT-3.5-turbo를 통해 답변을 생성합니다.

```python
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "파일을 선택해주세요."}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    lc_helper.process_pdf(pdf_file)
    return jsonify({"message": "PDF 파일이 성공적으로 업로드되었습니다."})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "질문을 입력해주세요."}), 400

    answer = lc_helper.get_answer(question)
    return jsonify({"answer": answer})
```

<br>

## 주요 개념 설명

#### GPT(혹은 언어 모델)에 대량의 데이터를 학습시키는 방법
일반적인 챗봇은 API를 통해 소수의 예시 데이터를 기반으로 학습하지만, 사내 규정이나 매뉴얼을 학습시키려면 대량의 데이터를 다루어야 합니다. 이를 위해 크게 두 가지 방법이 있습니다:
1. **RAG(검색 증강 생성)**: 미리 지정한 텍스트를 데이터베이스에 저장한 후 사용자가 입력한 질문과 관련된 정보를 검색하여 프롬프트에 추가, 보다 정확한 답변을 생성하는 기법입니다.
2. **파인튜닝**: 모델 자체에 대량의 데이터를 학습시키는 방식으로, 모델이 특정 도메인에 최적화된 성능을 발휘할 수 있도록 도와줍니다. 파인튜닝은 높은 연산 능력과 비용이 요구되기도 합니다.

#### 벡터 데이터 임베딩
RAG를 구현하려면 대량의 텍스트 데이터를 벡터 형태로 변환해야 합니다. 벡터는 텍스트 데이터를 숫자 배열로 표현한 것으로, 의미적으로 유사한 단어는 다차원 공간상에서 가까운 위치에 배치됩니다. 예를 들어, '개'와 '고양이'는 유사한 개념이므로 벡터 상에서 가까운 위치에 존재합니다. 반면, '개'와 '냉장고'는 연관성이 적어 멀리 떨어져 있습니다.

#### 벡터 DB
벡터 DB는 임베딩된 벡터 데이터를 저장하고 검색하는 시스템입니다. 질문에 대한 적절한 답변을 빠르게 생성하기 위해 벡터 DB에서 유사한 벡터를 검색하여 관련 정보를 찾아냅니다.

#### LangChain
LangChain은 대규모 문서를 효율적으로 처리하는 도구로, 벡터 DB와 통합되어 GPT-3.5-turbo와 함께 사용될 때 더욱 강력한 성능을 발휘합니다. 이를 통해 대량의 데이터를 벡터화해 저장하고, 사용자의 질문에 맞는 정보를 빠르게 검색하여 정확한 답변을 제공합니다. 또한, LangChain은 대화 기록을 관리해 문맥을 반영한 자연스러운 대화를 가능하게 합니다. GPT-3.5-turbo만 사용할 때보다 효율성과 정확성이 크게 향상됩니다.

<br>

## RAG 활용하여 보안 메뉴얼(PDF 파일)에 기반한 답변을 이끌어내는 방법
1. 보안 요원이 숙지해야 할 아파트 보안 매뉴얼을 벡터화하여 벡터 DB에 저장합니다.
2. 보안요원(사용자)가 "101동에 불이 났는데, 나는 어떻게 대응해야 할까?"라는 질문을 입력합니다.
3. 프로그램이 다음과 같은 처리를 수행합니다:
   a. "101동에 불이 났는데, 나는 어떻게 대응해야 할까?"라는 질문을 벡터화합니다.
   b. 벡터화된 보안 매뉴얼과 질문 문장을 비교하여, 관련된 문장을 검색합니다. 예를 들어, 매뉴얼의 "화재 발생 시 즉시 소방서에 신고 후, 비상구로 주민 유도"와 같은 정보를 찾습니다.
4. 검색된 보안 매뉴얼 정보를 프롬프트에 삽입해 GPT 모델에 전달하여 답변을 생성합니다.
5. 생성된 답변은 사용자가 화재 시 어떻게 대응해야 하는지 매뉴얼에 근거한 구체적인 지침을 제공합니다.

<br>

## 프로그램 비교

### 표 1: DialoGPT 프로그램 (이 프로그램이 궁금하다면, [여기](https://github.com/kks0507/text_embedding_DialoGPT.git)에서 확인할 수 있습니다.)

| **항목**                | **DialoGPT 프로그램**      
|------------------------|---------------------------
| **모델 사용**           | DialoGPT (GPT-2 기반)     
| **데이터 처리**         | CSV 파일 기반 처리        
| **확장성**              | 제한적 (고정된 CSV 파일)  
| **대화의 자연스러움**    | 문맥 이해가 부족           
| **데이터 업데이트**     | 수동 업데이트 필요        
| **검색 속도**           | 소규모 데이터셋에서만 빠름 
| **추론 정확도**         | 문맥 유사성 검색 어려움    

<br>

### 표 2: LangChain + GPT-3.5-turbo 프로그램

| **항목**                | **LangChain + GPT-3.5-turbo 프로그램** 
|------------------------ |----------------------------------------
| **모델 사용**           | GPT-3.5-turbo                          
| **데이터 처리**         | PDF 파일 임베딩 및 벡터 DB 사용          
| **확장성**              | 다양한 파일 형식 지원 및 실시간 처리      
| **대화의 자연스러움**   | GPT-3.5로 문맥 이해 및 응답 정확도 향상   
| **데이터 업데이트**     | 실시간 업데이트 가능                     
| **검색 속도**           | 대규모 데이터에서도 빠른 검색              
| **추론 정확도**         | 벡터 DB를 통한 정확한 문맥 검색         


<br>

## Contributor
- kks0507

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## Repository
프로젝트의 최신 업데이트는 [여기](https://github.com/kks0507/Langchain_PDF_ChatBot.git)에서 확인할 수 있습니다.
