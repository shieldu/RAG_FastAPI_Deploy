from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_utils import LangChainHelper  # langchain_utils.py에서 PDF 처리 클래스 가져오기
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 업로드 폴더 생성
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# LangChainHelper 인스턴스 생성
lc_helper = LangChainHelper()

# HTML 렌더링 엔드포인트
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("langchain.html", {"request": request})

# PDF 파일 업로드 API
@app.post("/upload_pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    if not pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일을 선택해주세요.")

    file_path = os.path.join("uploads", pdf.filename)
    
    # 파일 저장
    try:
        with open(file_path, "wb") as file:
            file.write(await pdf.read())
    except Exception as e:
        return JSONResponse(content={"message": f"파일을 저장하는 중 오류가 발생했습니다: {str(e)}"}, status_code=500)

    # 파일 크기 확인
    file_size = os.path.getsize(file_path)
    print(f"File saved at: {file_path}")
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        return JSONResponse(content={"message": "파일이 비어 있습니다."}, status_code=400)
    
    # PDF 처리
    try:
        lc_helper.process_pdf(file_path)  # 동기 함수로 호출
    except Exception as e:
        return JSONResponse(content={"message": f"파일을 처리하는 중 오류가 발생했습니다: {str(e)}"}, status_code=500)

    return JSONResponse(content={"message": "PDF 파일이 성공적으로 업로드되었습니다."})

# 질문을 통해 답변을 받는 API
@app.post("/ask_question")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return JSONResponse(content={"message": "질문을 입력해주세요."}, status_code=400)

    try:
        answer = await lc_helper.get_answer(question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(content={"message": f"질문에 대한 답변을 생성하는 중 오류가 발생했습니다: {str(e)}"}, status_code=500)

# FastAPI 애플리케이션 실행
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
