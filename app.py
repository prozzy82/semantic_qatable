import os
import streamlit as st
import datetime

st.set_page_config(page_title="Поиск в оглавлении", layout="wide")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import qdrant_client
import re
from sentence_transformers import CrossEncoder

# Загрузка переменных окружения
load_dotenv(dotenv_path=".env")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "enciclop"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# 1. СОЗДАЕМ ЭКЗЕМПЛЯР МОДЕЛИ ЭМБЕДДИНГОВ
embeddings_instance = get_embeddings()

@st.cache_resource
def get_qdrant_client():
    return qdrant_client.QdrantClient(
        url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False
    )

# 2. СОЗДАЕМ ЭКЗЕМПЛЯР КЛИЕНТА QDRANT
qdrant_client_instance = get_qdrant_client()

@st.cache_resource
def get_vector_store():
    # Эта функция теперь будет видеть определенную ранее embeddings_instance
    return QdrantVectorStore(
        client=qdrant_client_instance,
        collection_name=COLLECTION_NAME,
        embedding=embeddings_instance, # Используем глобально определенную embeddings_instance
    )

# 3. СОЗДАЕМ ЭКЗЕМПЛЯР ВЕКТОРНОГО ХРАНИЛИЩА
vector_store = get_vector_store()

# Инициализация модели Cross-Encoder для переранжирования
@st.cache_resource
def get_reranker():
    model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    try:
        model = CrossEncoder(model_name, device='cpu')
        return model
    except Exception as e:
        st.error(f"Не удалось загрузить модель reranker: {e}")
        return None

reranker = get_reranker()

# ... (остальная часть вашего кода: extract_structured_info, find_relevant_situations, интерфейс Streamlit) ...
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

        if section_candidate: section = section_candidate
        if point_candidate: point = point_candidate
        
        if title_candidate:
            title = title_candidate
        elif point_candidate:
            title = f"Наименование для пункта {point_candidate} не извлечено."
    else:
        fallback_pattern = re.compile(
            r"Пункт оглавления:\s*(?P<point_num>\d[\d\.]*[\d])\s*\.?"
            r"\s*(?P<title>.*)", re.IGNORECASE | re.DOTALL
        )
        fallback_match = fallback_pattern.search(page_content_text)
        if fallback_match:
            point_candidate = fallback_match.group("point_num").strip()
            title_candidate = fallback_match.group("title").strip()
            if point_candidate: point = point_candidate
            if title_candidate:
                title = title_candidate
            elif point_candidate:
                 title = f"Наименование для пункта {point_candidate} не извлечено."
    return {
        "section": section, "point": point, "title": title, "source": source,
        "score": 0, "full_content": page_content_text
    }

# Функция для поиска релевантных ситуаций с переранжированием
def find_relevant_situations(query, initial_top_k=20, final_top_k=7):
    docs_with_scores_qdrant = vector_store.similarity_search_with_score(query, k=initial_top_k)
    
    if not docs_with_scores_qdrant:
        return []

    if reranker is None:
        st.warning("Модель переранжирования не загружена. Результаты могут быть менее точными.")
        results_no_rerank = []
        for doc, score in docs_with_scores_qdrant[:final_top_k]:
            info = extract_structured_info(doc.page_content, doc.metadata)
            info["score"] = score 
            results_no_rerank.append(info)
        return results_no_rerank

    sentence_pairs = []
    candidate_docs_map = {} 
    for idx, (doc, qdrant_score) in enumerate(docs_with_scores_qdrant):
        sentence_pairs.append([query, doc.page_content])
        candidate_docs_map[idx] = (doc, qdrant_score)

    try:
        reranker_scores = reranker.predict(sentence_pairs, convert_to_tensor=True)
    except Exception as e:
        st.error(f"Ошибка при переранжировании: {e}")
        results_fallback = []
        for doc, score in docs_with_scores_qdrant[:final_top_k]:
            info = extract_structured_info(doc.page_content, doc.metadata)
            info["score"] = score 
            results_fallback.append(info)
        return results_fallback

    reranked_docs_with_new_scores = []
    for i in range(len(sentence_pairs)):
        original_doc, original_qdrant_score = candidate_docs_map[i]
        info = extract_structured_info(original_doc.page_content, original_doc.metadata)
        info["score"] = reranker_scores[i].item() 
        reranked_docs_with_new_scores.append(info)
        
    reranked_docs_with_new_scores.sort(key=lambda x: x["score"], reverse=True)
    
    return reranked_docs_with_new_scores[:final_top_k]


# --- Интерфейс Streamlit ---
st.title("🔍 Поиск в оглавлении")
st.write("Опишите вашу ситуацию, и система найдет соответствующие наименования пунктов спорных ситуаций.")

query = st.text_area("Опишите вашу ситуацию:", 
                     placeholder="Пример: нужно ли каждый месяц составлять акты оказанных услуг в рамках договора аренды",
                     height=150)

if "current_date" not in st.session_state:
    st.session_state.current_date = datetime.date.today().strftime("%d.%m.%Y")

if st.button("Найти ситуации"):
    if not query:
        st.warning("Пожалуйста, опишите вашу ситуацию")
    else:
        with st.spinner("Идет поиск и анализ релевантных наименований..."):
            situations = find_relevant_situations(query, initial_top_k=25, final_top_k=7) 
            
            if not situations:
                st.error("Не найдено релевантных наименований пунктов для вашего запроса. Попробуйте переформулировать запрос.")
            else:
                st.success(f"Найдено {len(situations)} наиболее релевантных наименований пунктов:")
                
                for i, sit in enumerate(situations, 1):
                    st.markdown(f"**{i}. {sit['title']}**")
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
                    if i < len(situations):
                        st.divider()
                    else:
                        st.markdown("<br>", unsafe_allow_html=True)

# Информация о системе
st.sidebar.title("О системе")
st.sidebar.info(
    """
    RAG + семантический поиск и переранжирование.
    """
)
st.sidebar.divider()
st.sidebar.markdown(f"""
<div style='font-size: 0.875em; color: gray;'>
    © Prozorovskiy Dmitriy.
    Дата: 01.03.2025
</div>
""", 
unsafe_allow_html=True)
