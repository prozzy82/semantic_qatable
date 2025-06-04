import os
import streamlit as st
import datetime

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

# Функция для извлечения структурированной информации
def extract_structured_info(page_content_text, metadata):
    source = metadata.get("source", "Источник не указан")
    section = "Раздел не определен"
    point = "Пункт не определен"
    title = page_content_text 

    pattern = re.compile(
        r"Раздел:\s*(?P<section>.*?)\s*"
        r"Пункт оглавления:\s*(?P<point_num>\d[\d\.]*[\d])\s*\.?"
        r"\s*(?P<title>.*)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(page_content_text)

    if match:
        section_candidate = match.group("section").strip()
        point_candidate = match.group("point_num").strip()
        title_candidate = match.group("title").strip()

        if section_candidate:
            section = section_candidate
        if point_candidate:
            point = point_candidate
        
        if title_candidate:
            title = title_candidate
        elif point_candidate:
            title = f"Наименование для пункта {point_candidate} не извлечено (исходный текст: '{page_content_text[:50]}...')."
            
    else:
        fallback_pattern = re.compile(
            r"Пункт оглавления:\s*(?P<point_num>\d[\d\.]*[\d])\s*\.?"
            r"\s*(?P<title>.*)",
            re.IGNORECASE | re.DOTALL
        )
        fallback_match = fallback_pattern.search(page_content_text)
        if fallback_match:
            point_candidate = fallback_match.group("point_num").strip()
            title_candidate = fallback_match.group("title").strip()

            if point_candidate:
                point = point_candidate
            if title_candidate:
                title = title_candidate
            elif point_candidate:
                 title = f"Наименование для пункта {point_candidate} не извлечено (исходный текст: '{page_content_text[:50]}...')."
        # Если и основной и fallback не сработали, title останется page_content_text

    return {
        "section": section,
        "point": point,
        "title": title,
        "source": source,
        "score": 0, # Инициализируем, будет перезаписано в find_relevant_situations
        "full_content": page_content_text
    }

# Функция для поиска релевантных ситуаций
def find_relevant_situations(query, top_k=5):
    docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    results = []
    for doc, score in docs_with_scores:
        info = extract_structured_info(doc.page_content, doc.metadata)
        info["score"] = score
        results.append(info)
    
    return results

# --- Интерфейс Streamlit ---
st.title("🔍 Поиск спорных налоговых ситуаций")
st.write("Опишите вашу ситуацию, и система найдет соответствующие наименования пунктов спорных ситуаций.")

query = st.text_area("Опишите вашу ситуацию:", 
                     placeholder="Пример: Можно ли учесть расходы на такси для сотрудника в командировке...",
                     height=150)

if "current_date" not in st.session_state:
    st.session_state.current_date = datetime.date.today().strftime("%d.%m.%Y")

if st.button("Найти ситуации"):
    if not query:
        st.warning("Пожалуйста, опишите вашу ситуацию")
    else:
        with st.spinner("Идет поиск релевантных наименований..."):
            situations = find_relevant_situations(query, top_k=5)
            
            if not situations:
                st.error("Не найдено релевантных наименований пунктов для вашего запроса. Попробуйте переформулировать запрос.")
            else:
                st.success(f"Найдено {len(situations)} релевантных наименований пунктов:")
                
                for i, sit in enumerate(situations, 1):
                    # Основное наименование пункта
                    st.markdown(f"**{i}. {sit['title']}**")
                    
                    # Дополнительная информация: Раздел, Пункт, Соответствие
                    # Используем HTML для более тонкого контроля над стилем и отступами
                    details_html = f"""
                    <div style="margin-left: 20px; font-size: 0.9em; color: #4F4F4F;">
                        <i>
                            Раздел: {sit['section']} • 
                            Пункт: {sit['point']} • 
                            Соответствие: {sit['score']:.2f}
                        </i>
                    </div>
                    """
                    st.markdown(details_html, unsafe_allow_html=True)
                    
                    # Добавляем разделитель между элементами, кроме последнего
                    if i < len(situations):
                        st.divider()
                    else:
                        st.markdown("<br>", unsafe_allow_html=True) # Небольшой отступ после последнего элемента

# Информация о системе
st.sidebar.title("О системе")
st.sidebar.info(
    """
   testing
    """
)
st.sidebar.divider()
st.sidebar.markdown(f"""
<div style='font-size: 0.875em; color: gray;'>
    © Prozorovskiy Dmitriy.
    Дата: {st.session_state.current_date}
</div>
""", 
unsafe_allow_html=True)
