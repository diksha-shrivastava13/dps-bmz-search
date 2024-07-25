import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse

# Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
az_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
az_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# models
embed_model = OpenAIEmbedding(embed_batch_size=10)
llm = AzureOpenAI(engine="gpt-4o", model="gpt-4o", temperature=0.0,
                  api_key=az_openai_api_key, azure_endpoint=az_openai_endpoint)

# Configurations
Settings.embed_model = embed_model
Settings.llm = llm


# File Processing Functions
def create_index(file_path):
    parser = LlamaParse(result_type="markdown", num_workers=8)
    documents = parser.load_data(file_path)
    index = VectorStoreIndex.from_documents(documents)

    return index


def recommended_fields_generation(index):
    processing_container.markdown("""<p style="color: #3ae2a5;">Almost done...</p>""", unsafe_allow_html=True)
    query_engine = index.as_query_engine()
    recommended_fields_data = query_engine.query(
        "Now that you have a good look at the document, as a data analyst who needs to provide enough insight and data"
        "to your manager so they can make the correct data-driven decisions on all factors, what information and fields"
        " would be the most useful to you? Make a list of all such fields and give your reasons on why it would be"
        "useful in order for you to assist in making data driven decisions and always have insights into how the"
        "program is processing and drive it to success. Respond in German."
    )
    st.markdown(recommended_fields_data)
    return recommended_fields_data


def user_query_answer(index, user_query):
    query_engine = index.as_query_engine()
    query_answer = query_engine.query(user_query + " Make the answer as detailed and as comprehensive as required."
                                      + " Make sure to use the documents as context to answer the question."
                                      + " If you cannot find an answer from the documents, tell the user to go through"
                                        "the original document they uploaded. Respond in German. Clearly mention all"
                                        "the numeric data related to the user query and the context associated with"
                                        "it. Mentioning numbers, if required, is very important to the user. Be as"
                                        "detailed as possible and include all relevant information about the question.")
    return query_answer


def display_information_once():
    file_path = "temp_file.pdf"
    if uploaded_file is not None:
        # time.sleep(5)
        processing_container.markdown("""<p style="color: #3ae2a5;">Creating Index...</p>""", unsafe_allow_html=True)
        query_index_ = create_index(file_path)

        recommended_fields_generation(query_index_)

        processing_container.markdown("""<p style="color: #3ae2a5;">Processing complete!</p>""", unsafe_allow_html=True)


st.set_page_config(
    page_title="PortFolio Navigator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
st.markdown("<h1 style='text-align: center;'>📈 Portfolio Navigator: Search </h1>", unsafe_allow_html=True)
st.markdown(" ")
st.markdown(
    "<p style='text-align: center;'>Upload your Document to Portfolio Navigator and get overview, insights, "
    "history and analysis.", unsafe_allow_html=True)

# Upload functionality
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
processing_container = st.empty()
processing_container.text(" ")

input_search_query = st.text_input("Search:", value=st.session_state.get('search_query', ''),
                                   key="search_query")

if uploaded_file is not None:
    temp_filepath = Path("temp_file.pdf")
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.read())
    processing_container.markdown("File uploaded...")
    time.sleep(2)
    processing_container.markdown("""<p style="color: #3ae2a5;">Enter the search query</p>""", unsafe_allow_html=True)
    # display_information_once()
else:
    st.markdown("""<p style="color: #3ae2a5;">Please upload a PDF file.</p>""", unsafe_allow_html=True)


# Conditionally update the session state if the input has changed
if 'search_query' not in st.session_state or input_search_query != st.session_state['search_query']:
    uploaded_file = None
    st.session_state['search_query'] = input_search_query


# Perform the search operation using the search term from session state
if st.session_state['search_query']:
    answer_processing_container = st.empty()
    answer_processing_container.markdown("""<p style="color: #3ae2a5;">Processing request...</p>""",
                                         unsafe_allow_html=True)
    filepath = "temp_file.pdf"
    query_index = create_index(filepath)
    answer = user_query_answer(query_index, st.session_state['search_query'])
    if answer:
        answer_processing_container.markdown("""<p style="color: #3ae2a5;">Found results!</p>""",
                                             unsafe_allow_html=True)
        st.markdown(answer)
    else:
        st.markdown("No answer found for your query.")
