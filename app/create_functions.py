#%%
import glob
import os
from typing import List
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

#%%
def create_chat(t=0.0):
    """
    Configura os objetos do AzureChatOpenAI para serem utilizados.

    Args:
        t (float): O valor da temperatura para determinar o comportamento da resposta, por padrão é 0.0.

    Returns:
        AzureChatOpenAI: O objeto de AzureChatOpenAi configurado.
    """
    llm_chat = AzureChatOpenAI(
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_version=os.getenv('OPENAI_API_VERSION'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_type=os.getenv('OPENAI_API_TYPE'),
        deployment_name=os.getenv('DEPLOYMENT_NAME'),
        temperature=t,
    )
    return llm_chat


# %%
def create_embeddings(chunk_size=1):
    """
    Configura os objetos do OpenAIEmbeddings para serem utilizados.

    Args:
        chunk_size (int): Quantidade de tokens em cada chunk.

    Returns:
        embeddings (OpenAIEmbeddings): O objeto de OpenAIEmbeddings configurado.
    """
    embeddings = OpenAIEmbeddings(
        deployment=os.getenv('EMBEDDING_DEPLOYMENT_NAME'),
        chunk_size=chunk_size,
    )
    return embeddings


# %%
def define_pastas(data=None):
    """
    Definir caminho da pasta de dados.

    Args:
        data (str): Caminho da pasta de dados.

    Returns:
        str: Caminho da pasta de dados.
    """
    data = data or r'C:\Users\BlueShift\Downloads\read_pdf\data'

    return data


# %%
def verify_pdf(folder: str) -> List[str]:
    """
    Verifica se a pasta existe e retorna os nomes dos arquivos PDF na pasta.

    Args:
        pasta(str): Caminho da pasta.

    Return:
        List[str]: Lista com os nomes dos arquivos PDF na pasta.

    Raises:
        FileNotFoundError: Se a pasta não existir.
        ValueError: Se não houver arquivos PDF na pasta.
    """
    try:
        if not os.path.exists(folder):
            raise FileNotFoundError(f'A pasta {folder} não existe')

        files_folder = glob.glob(os.path.join(folder, '*.pdf'))
        if not files_folder:
            raise ValueError('Erro: Não há arquivos .pdf na pasta')
        else:
            # Para cada arquivo na pasta, retornar apenas o nome sem o caminho completo
            nomes_arquivos = [
                os.path.basename(arquivo) for arquivo in files_folder
            ]

        return nomes_arquivos
    except (FileNotFoundError, ValueError) as e:
        raise
