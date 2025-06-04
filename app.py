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
    """Извлекает раздел, пункт и название из page_content и source из metadata"""
    source = metadata.get("source", "Источник не указан")
    # Значения по умолчанию
    section = "Раздел не определен"
    point = "Пункт не определен"
    title = page_content_text # В худшем случае, если ничего не распарсится

    # Основной regex для извлечения структурированных данных
    # Ожидаемый формат: "Раздел: [ТЕКСТ_РАЗДЕЛА] Пункт оглавления: [НОМЕР_ПУНКТА опц_точка] [ТЕКСТ_НАЗВАНИЯ_ПУНКТА]"
    pattern = re.compile(
        r"Раздел:\s*(?P<section>.*?)\s*"
        r"Пункт оглавления:\s*(?P<point_num>\d[\d\.]*[\d])\s*\.?" # Номер пункта, может содержать точки, заканчивается цифрой
        r"\s*(?P<title>.*)", # Все остальное - название, включая пробелы после точки, если они есть
        re.IGNORECASE | re.DOTALL # DOTALL чтобы .* захватывал и переносы строк
    )
    match = pattern.search(page_content_text)

    if match:
        section_candidate = match.group("section").strip()
        point_candidate = match.group("point_num").strip()
        title_candidate = match.group("title").strip() # .strip() уберет лишние пробелы вокруг названия

        if section_candidate:
            section = section_candidate
        if point_candidate:
            point = point_candidate
        
        if title_candidate: # Если что-то было захвачено как title и оно не пустое после strip
            title = title_candidate
        elif point_candidate: # Если title пуст (или только пробелы), но есть пункт
            title = f"Наименование для пункта {point_candidate} не извлечено (возможно, отсутствует в исходном тексте)."
        # Если и title_candidate пуст, и point_candidate не определен (хотя regex требует point_num),
        # то title останется page_content_text.
            
    else:
        # Если основной regex не сработал, пробуем извлечь хотя бы пункт и название без раздела
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
            elif point_candidate: # Title пуст, но пункт есть
                 title = f"Наименование для пункта {point_candidate} не извлечено (возможно, отсутствует в исходном тексте)."
            # Раздел в этом случае останется "Раздел не определен" или можно попытаться найти его отдельно,
            # но это усложнит логику без гарантии успеха, если формат сильно варьируется.

    return {
        "section": section,
        "point": point,
        "title": title, # Это "наименование пункта спорной ситуации"
        "source": source,
        "score": 0, # Инициализируем, будет перезаписано позже
        "full_content": page_content_text
    }

# Функция для поиска релевантных ситуаций
def find_relevant_situations(query, top_k=5):
    """Поиск релевантных спорных ситуаций по запросу"""
    docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    results = []
    for doc, score in docs_with_scores:
        info = extract_structured_info(doc.page_content, doc.metadata)
        info["score"] = score # Оценка релевантности от Qdrant
        results.append(info)
    
    # Qdrant обычно возвращает уже отсортированными по score (от большего к меньшему для similarity)
    # Если нет, можно добавить results.sort(key=lambda x: x["score"], reverse=True)
    return results

# --- Интерфейс Streamlit ---
st.title("🔍 Поиск релевантных спорных налоговых ситуаций")
st.write("Опишите вашу ситуацию, и система найдет соответствующие наименования пунктов спорных ситуаций.")

# Поле ввода запроса
query = st.text_area("Опишите вашу ситуацию:", 
                     placeholder="Пример: Можно ли учесть расходы на такси для сотрудника в командировке...",
                     height=150)

if "current_date" not in st.session_state:
    st.session_state.current_date = datetime.date.today().strftime("%d.%m.%Y")

# Кнопка поиска
if st.button("Найти ситуации"):
    if not query:
        st.warning("Пожалуйста, опишите вашу ситуацию")
    else:
        with st.spinner("Идет поиск релевантных наименований..."):
            situations = find_relevant_situations(query, top_k=5) # Можете изменить top_k
            
            if not situations:
                st.error("Не найдено релевантных наименований пунктов для вашего запроса. Попробуйте переформулировать запрос.")
            else:
                st.success(f"Найдено {len(situations)} релевантных наименований пунктов:")
                
                # Выводим только нумерованный список наименований пунктов
                for i, sit in enumerate(situations, 1):
                    # sit['title'] теперь должно содержать корректно извлеченное наименование пункта
                    st.markdown(f"{i}. **{sit['title']}**")
                    # Если хотите добавить немного больше информации (например, номер пункта и релевантность):
                    # st.caption(f"   (Пункт документа: {sit['point']}, Релевантность: {sit['score']:.2f})")


# Информация о системе
st.sidebar.title("О системе")
st.sidebar.info(
    """
    Эта система выполняет поиск релевантных наименований
    пунктов спорных налоговых ситуаций из базы знаний.
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
