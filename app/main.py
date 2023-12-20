#%%
# Importar as funções do arquivo de funções
from create_functions import create_chat, create_embeddings, define_pastas, verify_pdf
from langchain import VectorDBQA
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
#from python-dotenv import load_dotenv

#%%
#load_dotenv()

#%%
def main(query=None):
    """
+    Essa função processa um arquivo PDF, extrai suas páginas, cria um banco de dados de embeddings
+    e realiza uma busca de perguntas e respostas com base na consulta fornecida.
+    """
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

    # Criar o banco de dados
    db = Chroma.from_documents(pages, embeddings)

    # Cria o objeto QA(Question Answering)
    # qa = VectorDBQA.from_chain_type(llm,chain_type="stuff",vectorstore=db)
    qa2 = RetrievalQA.from_chain_type(
        llm, chain_type='stuff', retriever=db.as_retriever()
    )
    # Executar a consulta
    query = query or 'Do que se trata as informações deste arquivo?'

    print(qa2.run(query))

main("O que é docker?")
# %%
