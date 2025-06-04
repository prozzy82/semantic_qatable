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
        embedding=embeddings,
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

# Функция для извлечения структурированной информации
def extract_structured_info(page_content_text, metadata):
    """Извлекает раздел, пункт и название из page_content и source из metadata"""
    source = metadata.get("source", "Источник не указан")
    section = "Не определен"
    point = "Не определен"
    # "Наименование пункта" мы будем считать то, что извлекается как title
    title = page_content_text # По умолчанию, если парсинг не удался

    # Пример вашего формата: "Раздел: Б1. Расходы ... Пункт оглавления: 24. Учет расходов ..."
    match = re.search(
        r"Раздел:\s*(?P<section>[^П]+?)\s*Пункт оглавления:\s*(?P<point_num>\d+(\.\d+)*)\.?\s*(?P<title>.+)",
        page_content_text,
        re.IGNORECASE
    )
    if match:
        section = match.group("section").strip()
        point = match.group("point_num").strip()
        title_candidate = match.group("title").strip()
        # Проверяем, не начинается ли title с номера пункта, и если да, то удаляем его
        if title_candidate.lower().startswith(point.lower()):
             title = title_candidate[len(point):].strip(". ")
        else:
             title = title_candidate
    elif page_content_text:
        # Альтернативные, более простые regex, если основной не сработал
        point_match = re.search(r"Пункт оглавления[:\s]*(\d+(\.\d+)*)", page_content_text, re.IGNORECASE)
        title_match = re.search(r"Пункт оглавления[:\s]*\d+(\.\d+)*\.?\s*(.+)", page_content_text, re.IGNORECASE)

        if point_match and point_match.group(1):
            point = point_match.group(1).strip()
        if title_match and title_match.group(1):
            title = title_match.group(1).strip()
        elif page_content_text:
            title = page_content_text[:150] + "..." if len(page_content_text) > 150 else page_content_text
        
        # Попробуем извлечь раздел отдельно, если он есть
        section_match = re.search(r"Раздел[:\s]*(.*?)(Пункт оглавления|$)", page_content_text, re.IGNORECASE | re.DOTALL)
        if section_match and section_match.group(1):
            section = section_match.group(1).strip().rstrip(',')

    return {
        "section": section,
        "point": point,
        "title": title, # Это будет "наименование пункта спорной ситуации"
        "source": source,
        "full_content": page_content_text
    }

# Функция для поиска релевантных ситуаций
def find_relevant_situations(query, top_k=5):
    """Поиск релевантных спорных ситуаций по запросу"""
    docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    results = []
    for doc, score in docs_with_scores:
        info = extract_structured_info(doc.page_content, doc.metadata)
        info["score"] = score
        results.append(info)
    
    results.sort(key=lambda x: x["score"], reverse=True) # Qdrant обычно возвращает уже отсортированными
    return results

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
st.sidebar.title("test")
st.sidebar.info("""

""")

st.sidebar.divider()
st.sidebar.markdown("""
<div style='font-size: 0.875em; color: gray;'>
    ©Prozorovskiy Dmitriy.
    Date: {current_date}
</div>
""".format(current_date=st.session_state.get("current_date", "01.03.2025")), 
unsafe_allow_html=True)
