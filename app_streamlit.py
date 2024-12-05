
import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import os


def analise_sentimento(avaliacao):
    instrucoes = f"""
    Analise o seguinte texto e determine se o sentimento geral é positivo, neutro ou negativo.
              
    • Avaliação:
    {avaliacao}
    """
    
    return modelo.invoke(instrucoes).content


def resumir_enredo(filme):
    instrucoes = f"""
    Resuma o enredo do seguinte filme, sem entregar surpresas do final do filme:

    {filme}
    """

    return modelo.invoke(instrucoes).content


api_key = api_key=os.getenv('GEMINI_KEY')
modelo = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key)

tool_analise_sentimento = Tool(
    name='Análise de Sentimento',
    func=analise_sentimento,
    description='Analisar se uma opinião sobre um filme é positiva, neutra ou negativa.'
)

tool_resumir_enredo = Tool(
    name='Resumir Enredo',
    func=resumir_enredo,
    description='Resume o enredo de um filme.'
)

if 'memoria' not in st.session_state:
    st.session_state['memoria'] = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

agente = initialize_agent(
    tools=[tool_analise_sentimento, tool_resumir_enredo],
    llm=modelo,
    agent='conversational-react-description',
    verbose=True,
    memory=st.session_state['memoria']
)

st.set_page_config(page_title='TP3 - Filmes com LangChain', page_icon='🎬')

st.title('Filmes com LangChain')

avatares = {
'human': 'user',
'ai': 'assistant'
}

if st.sidebar.button('Limpar histórico de mensagens'):
    st.session_state['memoria'].chat_memory.clear()

for mensagem in st.session_state['memoria'].chat_memory.messages:
    st.chat_message(avatares[mensagem.type]).write(mensagem.content)

if prompt := st.chat_input('Digite sua mensagem'):
    st.chat_message('user').write(prompt)
    with st.spinner('Processando...'):
        resposta = agente.run(prompt)
        st.chat_message('assistant').write(resposta)
