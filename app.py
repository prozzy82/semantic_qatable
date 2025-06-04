import os
import streamlit as st

# ВАЖНО: Эта строка должна быть первой командой Streamlit
st.set_page_config(page_title="Поиск спорных налоговых ситуаций", layout="wide")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
import qdrant_client
import re

# Загрузка переменных окружения
load_dotenv(dotenv_path=".env")
PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "enciclop"  # Имя вашей коллекции в Qdrant

# Инициализация модели эмбеддингов
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = get_embeddings()

# Инициализация клиента Qdrant
@st.cache_resource
def get_qdrant_client():
    client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False
    )
    return client

qdrant_client = get_qdrant_client()

# Инициализация векторного хранилища
@st.cache_resource
def get_vector_store():
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

vector_store = get_vector_store()

# Инициализация LLM для анализа результатов
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model_name="deepseek/deepseek-r1-0528",
        openai_api_key=PROVIDER_API_KEY,
        openai_api_base="https://api.novita.ai/v3/openai",
        temperature=0.3
    )

llm = get_llm()

# Функция для извлечения структурированной информации из метаданных
def extract_structured_info(metadata):
    """Извлекает раздел, пункт и название из метаданных документа"""
    # Пример формата: "Раздел 2.1, Пункт 2.1.3: Название ситуации"
    source = metadata.get("source", "")
    content = metadata.get("content", "")
    
    # Пытаемся извлечь структурированную информацию
    section = metadata.get("section", "")
    point = metadata.get("point", "")
    title = metadata.get("title", "")
    
    # Если в метаданных нет явных полей, пробуем парсить из текста
    if not section or not point:
        # Ищем паттерны типа "Раздел X.X" или "Пункт X.X.X"
        section_match = re.search(r"Раздел\s*([\d.]+)", content or "")
        point_match = re.search(r"Пункт\s*([\d.]+)", content or "")
        
        if section_match:
            section = section_match.group(1)
        if point_match:
            point = point_match.group(1)
    
    # Если название не найдено, используем начало контента
    if not title and content:
        title = content[:100] + "..." if len(content) > 100 else content
    
    return {
        "section": section,
        "point": point,
        "title": title,
        "source": source
    }

# Функция для поиска релевантных ситуаций
def find_relevant_situations(query, top_k=5):
    """Поиск релевантных спорных ситуаций по запросу"""
    # Поиск похожих документов в векторной базе
    docs = vector_store.similarity_search(query, k=top_k)
    
    # Извлечение структурированной информации
    results = []
    for doc in docs:
        info = extract_structured_info(doc.metadata)
        info["score"] = doc.metadata.get("score", 0)  # Оценка релевантности
        results.append(info)
    
    return results

# Функция для генерации отчета по найденным ситуациям
def generate_situation_report(situations, query):
    """Генерирует структурированный отчет с помощью LLM"""
    # Формируем контекст для LLM
    context = "Найдены следующие релевантные спорные ситуации:\n"
    for i, sit in enumerate(situations, 1):
        context += f"{i}. Раздел: {sit['section']}, Пункт: {sit['point']}, Название: {sit['title']}\n"
    
    # Промпт для LLM
    prompt = f"""
    Пользователь описал следующую ситуацию: 
    "{query}"
    
    {context}
    
    Проанализируй найденные спорные ситуации и составь отчет в следующем формате:
    1. Для каждой ситуации укажи:
       - Раздел и пункт
       - Краткое название ситуации
       - Почему она релевантна запросу пользователя (1-2 предложения)
    2. В конце добавь общие рекомендации по работе с этими ситуациями.
    
    Отвечай только на русском языке.
    """
    
    # Генерация ответа
    response = llm.invoke(prompt)
    return response.content

# --- Интерфейс Streamlit ---
st.title("🔍 Поиск релевантных спорных налоговых ситуаций")
st.write("Опишите вашу ситуацию, и система найдет соответствующие спорные вопросы в базе знаний")

# Поле ввода запроса
query = st.text_area("Опишите вашу ситуацию:", 
                     placeholder="Опишите спорную ситуацию из вашей практики...",
                     height=150)

# Кнопка поиска
if st.button("Найти релевантные ситуации"):
    if not query:
        st.warning("Пожалуйста, опишите вашу ситуацию")
    else:
        with st.spinner("Поиск релевантных ситуаций..."):
            # Поиск релевантных ситуаций
            situations = find_relevant_situations(query, top_k=5)
            
            if not situations:
                st.error("Не найдено релевантных ситуаций для вашего запроса")
            else:
                st.success(f"Найдено {len(situations)} релевантных ситуаций")
                
                # Показать сырые результаты поиска
                with st.expander("Просмотреть найденные ситуации"):
                    for i, sit in enumerate(situations, 1):
                        st.subheader(f"Ситуация #{i}")
                        st.write(f"**Раздел:** {sit['section']}")
                        st.write(f"**Пункт:** {sit['point']}")
                        st.write(f"**Название:** {sit['title']}")
                        st.write(f"**Источник:** {sit['source']}")
                        st.write(f"**Релевантность:** {sit['score']:.2f}")
                        st.divider()
                
                # Генерация аналитического отчета
                with st.spinner("Анализируем результаты..."):
                    report = generate_situation_report(situations, query)
                
                # Показать отчет
                st.subheader("Аналитический отчет по найденным ситуациям")
                st.markdown(report)

# Информация о системе
st.sidebar.title("О системе")
st.sidebar.info("""
Эта система анализирует описание спорных налоговых ситуаций и находит 
релевантные пункты в базе знаний спорных вопросов налогового законодательства.

**Как это работает:**
1. Вы описываете ситуацию из практики
2. Система ищет похожие ситуации в векторной базе знаний
3. LLM анализирует результаты и формирует отчет
""")

st.sidebar.divider()
st.sidebar.markdown("""
<div style='font-size: 0.875em; color: gray;'>
    ©Prozorovskiy Dmitriy.
    Date: {current_date}
</div>
""".format(current_date=st.session_state.get("current_date", "01.03.2025")), 
unsafe_allow_html=True)