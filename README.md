# Read a Pdf with LLM

Pacote para leitura de arquivo pdf e interação com suas informações

## Instalação e configuração

1. Clone o repositório:

> git clone https://github.com/MoisesArruda/Read_pdfLLM.git

> cd Read_pdf_llm

2. Configure a versão correta do Python com *pyenv*:

> pyenv local 3.10.10

3. Instale as dependências do projeto:

> poetry init

4. Ative o ambiente virtual:

> poetry shell

5. Crie um arquivo *.env* na raiz do projeto para armazenar sua OpenAI API key. Adicione suas chaves nesse arquivo:

> OPENAI_API_KEY=your_OPENAI_API_KEY
> OPENAI_API_BASE=OPENAI_API_BASE
> OPENAI_API_TYPE=OPENAI_API_TYPE
> OPENAI_API_VERSION=OPENAI_API_VERSION
> DEPLOYMENT_NAME=DEPLOYMENT_NAME
> EMBEDDING_DEPLOYMENT_NAME=EMBEDDING_DEPLOYMENT_NAME

6. Em *main.py* na variavel data chamando a função define_pastas(r"Nome da pasta") atualize o nome da sua pasta onde estará o arquivo pdf

> data = define_pastas(r"Nome da pasta")

7. Coloque o arquivo dentro da pasta

8. Execute os testes para garantir que tudo está funcionando como esperado:

> task test

9. Execute o comando para ver a documentação do projeto:

> task doc

