import streamlit as st
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA


class DiaryRAG:
    """PDF를 기반으로 질의응답을 수행하는 RAG 클래스"""
    def __init__(self, api_key: str, pdf_path: str):
        self.api_key = api_key
        self.pdf_path = pdf_path
        self.vectorstore = None

    def load_pdf(self) -> str:
        """PDF 파일에서 텍스트 추출"""
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def build_vectorstore(self, text: str):
        """텍스트를 청크로 분할 후 벡터스토어 생성"""
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.vectorstore = FAISS.from_documents(documents, embeddings)

    def answer_question(self, question: str) -> str:
        """질문에 대한 답변 생성"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore가 초기화되지 않았습니다.")
        llm = ChatOpenAI(model_name="gpt-4o", api_key=self.api_key)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=self.vectorstore.as_retriever())
        return qa.invoke({"query": question})["result"]


class DiaryApp:
    """Streamlit UI 클래스"""
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.api_key = None
        self.question = None

    def run(self):
        st.header("철수 일기 내용 물어보기")
        self.api_key = st.text_input("OPENAI API KEY를 입력하세요.", type="password")
        self.question = st.text_input("질문을 입력하세요.")

        if st.button("답변 확인"):
            if not self.api_key or not self.question:
                st.warning("API 키와 질문을 모두 입력해주세요.")
                return

            with st.spinner("일기 내용을 검토해 답변을 생성 중입니다..."):
                try:
                    rag = DiaryRAG(api_key=self.api_key, pdf_path=self.pdf_path)
                    text = rag.load_pdf()
                    rag.build_vectorstore(text)
                    answer = rag.answer_question(self.question)
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"오류 발생: {e}")


if __name__ == "__main__":
    app = DiaryApp(pdf_path="diary.pdf")
    app.run()