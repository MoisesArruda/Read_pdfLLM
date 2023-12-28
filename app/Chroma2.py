#%%
from langchain.document_loaders import PyPDFDirectoryLoader
from create_functions import create_chat, create_embeddings, define_pastas, verify_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import load_prompt

# %%
#Carrega a variavel de ambiente
load_dotenv()

#%%
'''class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})
'''

#def main(query=None):
"""
Essa função processa um arquivo PDF, extrai suas páginas, cria um banco de dados de embeddings
e realiza uma busca de perguntas e respostas com base na consulta fornecida.
"""
# Criar o objeto LLM
llm = create_chat()

# Criar o objeto Embeddings
embeddings = create_embeddings()

# Definir o caminho da pasta de dados
data = define_pastas()

# Verificar se a pasta existe e retornar os nomes dos arquivos PDF na pasta
try:
    arquivos = verify_pdf(data)[0]
except (FileNotFoundError, ValueError) as e:
    print(e)

# Carregar o arquivo PDF
loader = PyPDFLoader(f'{data}/{arquivos}')
pages = loader.load_and_split()

#%%
# Criar o banco de dados
db = Chroma.from_documents(pages, embeddings)
db

#%%
template = """You are a system who need to help students. Answer the question based on the information provided. If the
question cannot be answered using the information provided answer
with "I don't know"

Question: {query}
Context: {contexto}
Answer:"""

# %%
prompt = PromptTemplate(template=template,input_variables=['query','contexto'])
prompt

#%%
memory = ConversationBufferMemory(input_key="query",return_messages=True)
memory

# %%
query = 'O que é docker e  kubernets?'
contexto = db.similarity_search(query,k=2)
contexto = "---".join([doc.page_content for doc in contexto]).replace('\n','  ')
# %%
contexto
# %%
llm_chain= LLMChain(llm=llm,prompt=prompt,verbose=True)
# %%
llm_chain.run(query=query,contexto=contexto,memory=memory)

# %%
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    verbose=True,
)

# %%
qa_chain(query)
# %%
