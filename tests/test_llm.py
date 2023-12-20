import os 
import pytest
from app.create_functions import create_chat, create_embeddings, define_pastas, verify_pdf
import dotenv

def test_file_exists():
    """Testa se o arquivo .env existe."""
     
    assert os.path.exists('.env')


def test_create_chat_sucess():
    """Testa o comportamento da função `create_chat()` quando o arquivo .env existe."""

    dotenv.load_dotenv()

    openai_api_base=os.getenv('OPENAI_API_BASE'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    openai_api_type=os.getenv('OPENAI_API_TYPE'),
    deployment_name=os.getenv('DEPLOYMENT_NAME')

    # Chamar a função
    llm_chat = create_chat()

    # Verificar se a conexão foi bem-sucedida
    assert llm_chat is not None
