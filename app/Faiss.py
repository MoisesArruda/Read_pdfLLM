#%%
# Importar as funções do arquivo de funções
from create_functions import create_chat, create_embeddings, define_pastas, verify_pdf
from langchain import VectorDBQA
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

# %%
def main(query=None):
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

    # Criar o banco de dados
    db = Chroma.from_documents(pages, embeddings)

    # Cria o objeto QA(Question Answering)
    # qa = VectorDBQA.from_chain_type(llm,chain_type="stuff",vectorstore=db)
    qa2 = RetrievalQA.from_chain_type(
    llm, chain_type='stuff', retriever=db.as_retriever()
    )
    # Executar a consulta
    query = query or 'Do que se trata as informações deste arquivo?'

    output= qa2.run(query)
    print(output)

    return llm,db,query, output,qa2


def main2():
    # %%
    llm,db,query, output,qa2 = main()

    # %%
    print(llm,query,output,qa2)

    # %%
    FAISS_index = FAISS.from_documents(pages,embeddings)

    # %%
    docs = FAISS_index.similarity_search(query,k=2)
    for doc in docs:
        print(str(doc.metadata["page"])+ ":", doc.page_content[:300])

    # %%
    print(docs[0].page_content)

    # %%
    template = """You need to help students. Answer the question based on the information provided. If the
    question cannot be answered using the information provided answer
    with "I don't know"

    Question: {query}
    Chatbot:"""

    #%%
    query = "O que é Docker?"

    #%%
    prompt = PromptTemplate(template=template,input_variables=['query'])

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True,input_key="query")

    # %%
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    # %%
    llm_chain.run(query)
    # %%
    memory