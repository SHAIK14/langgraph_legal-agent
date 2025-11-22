from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

path = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
loader = PyPDFLoader(path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(documents=chunks,   embedding=embeddings, persist_directory="./chroma_db")

query = "What is the main contribution of the paper?"

results = vectorstore.similarity_search(query, k=3)


context = "\n\n".join([doc.page_content for doc in results])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions about the paper."),
    ("human", "{context}\n\nQuestion: {question}")
])


prompt = prompt_template.invoke({"context": context, "question": query})

response = model.invoke(prompt)

print(response)








