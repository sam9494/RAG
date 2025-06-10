from dataclasses import dataclass
from typing import Optional, List
from abc import ABC, abstractmethod
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """Configuration settings for the application."""
    # File paths
    text_path: str = os.getenv("TEXT_PATH", "material/hr_policy.txt")
    
    # Index names
    text_index_name: str = os.getenv("TEXT_INDEX_NAME", "faiss_index")
    
    # Document processing settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # OpenAI settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    model_name: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Validate file paths
        if not os.path.exists(self.text_path):
            logger.warning(f"Text file not found at {self.text_path}")

# Utility functions
def create_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings instance."""
    try:
        return OpenAIEmbeddings()
    except Exception as e:
        logger.error(f"Failed to create embeddings: {str(e)}")
        raise

def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """Create a text splitter with specified parameters."""
    try:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    except Exception as e:
        logger.error(f"Failed to create text splitter: {str(e)}")
        raise

def create_qa_chain(
    db: FAISS,
    temperature: float = 0,
    prompt: Optional[PromptTemplate] = None
) -> RetrievalQA:
    """Create a QA chain with the given vector store and optional custom prompt."""
    try:
        llm = ChatOpenAI(temperature=temperature)
        if prompt is None:
            template = """
            你是公司內部知識庫助理，請根據以下內容回答問題：

            內容：
            {context}

            問題：
            {question}

            ⚠️ 請只根據「內容」回答，不要使用你自己的知識。若無法回答請說「我找不到答案」。
            """
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    except Exception as e:
        logger.error(f"Failed to create QA chain: {str(e)}")
        raise

def save_faiss_index(db: FAISS, index_name: str) -> None:
    """Save FAISS index to local storage."""
    try:
        db.save_local(index_name)
        logger.info(f"Successfully saved FAISS index to {index_name}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {str(e)}")
        raise

def load_faiss_index(index_name: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """Load FAISS index from local storage."""
    try:
        if not os.path.exists(index_name):
            raise FileNotFoundError(f"FAISS index not found at {index_name}")
        return FAISS.load_local(index_name, embeddings)
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {str(e)}")
        raise

def query_qa_chain(qa_chain: RetrievalQA, query: str) -> None:
    """Query the QA chain and print results."""
    try:
        result = qa_chain.invoke({"query": query})
        print("\nAI 回答：")
        print(result["result"])
        print("\n參考資料：")
        for doc in result["source_documents"]:
            print(f"\n{doc.page_content}")
    except Exception as e:
        logger.error(f"Failed to query QA chain: {str(e)}")
        raise

# Base processor
class BaseDocumentProcessor(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.config.validate()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def load_documents(self):
        """Load documents from the source file."""
        pass
        
    def process(self):
        """Main processing pipeline."""
        try:
            # Load documents
            self.logger.info(f"Loading documents from {self.get_source_path()}")
            documents = self.load_documents()
            
            # Split documents
            self.logger.info("Splitting documents into chunks")
            text_splitter = create_text_splitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            texts = text_splitter.split_documents(documents)
            
            # Create embeddings
            self.logger.info("Creating embeddings")
            embeddings = create_embeddings()
            
            # Create FAISS index
            self.logger.info("Creating FAISS index")
            db = FAISS.from_documents(texts, embeddings)
            
            # Save index
            self.logger.info(f"Saving FAISS index to {self.get_index_name()}")
            save_faiss_index(db, self.get_index_name())
            
            # Create QA chain
            self.logger.info("Creating QA chain")
            qa_chain = create_qa_chain(
                db,
                temperature=self.config.temperature,
                prompt=self.get_prompt_template()
            )
            
            return qa_chain
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            raise
            
    @abstractmethod
    def get_source_path(self) -> str:
        """Get the path to the source file."""
        pass
        
    @abstractmethod
    def get_index_name(self) -> str:
        """Get the name for the FAISS index."""
        pass
        
    def get_prompt_template(self) -> PromptTemplate:
        """Get the prompt template for QA chain."""
        template = """
你是公司內部知識庫助理，請根據以下內容回答問題。

1. 如果問題是詢問「各種假有幾天」或「報假可以幾天」這類總結性問題，請只列出每種假別的天數（如：特休假、病假、婚假等），不要提供詳細規則或申請流程。
2. 如果問題是針對某一種假的詳細規則、申請流程或證明文件，才提供該假的詳細資訊。
3. 請用條列式回答。
4. 只根據「內容」回答，不要使用你自己的知識。若無法回答請說「我找不到答案」。

內容：
{context}

問題：
{question}
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
    def test_qa_chain(self, qa_chain, query: str = "報假可以幾天？"):
        """Test the QA chain with a query."""
        self.logger.info(f"Testing QA chain with query: {query}")
        query_qa_chain(qa_chain, query)

# Text processor
class TextProcessor(BaseDocumentProcessor):
    def load_documents(self):
        """Load documents from text file."""
        loader = TextLoader(self.get_source_path(), encoding="utf-8")
        return loader.load()
        
    def get_source_path(self) -> str:
        """Get the path to the text file."""
        return self.config.text_path
        
    def get_index_name(self) -> str:
        """Get the name for the text FAISS index."""
        return self.config.text_index_name

def main():
    config = Config()
    processor = TextProcessor(config)
    qa_chain = processor.process()
    processor.test_qa_chain(qa_chain)

if __name__ == "__main__":
    main() 