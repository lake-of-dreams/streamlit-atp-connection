import os
from threading import Thread
from typing import cast

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader, OracleTextSplitter
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.utilities import OracleSummary
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from atp_connection import ATPConnection

load_dotenv()
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #af4f3b;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #f66648;
    color:#ffffff;
    }
div.stFormSubmitButton > button:first-child {
    background-color: #af4f3b;
    color:#ffffff;
}
div.stFormSubmitButton > button:hover {
    background-color: #f66648;
    color:#ffffff;
    }
</style>""", unsafe_allow_html=True)

default_query = "select 1 from dual"
default_char_col_size = 1000
state = st.session_state


class WorkerThread(Thread):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        self.func()


def collect_data() -> None:
    state.stage = collect_data
    with st.form("main_page", clear_on_submit=True, border=0):
        st.form_submit_button("Connect to an ATP", on_click=start)
    state.completed_stage = collect_data


def start() -> None:
    state.stage = start
    with st.form("compartment_id_form", clear_on_submit=True, border=0):
        st.text_input(label="Compartment ID", value=os.environ.get("default_compartment_id"), key="compartment_id")
        st.form_submit_button(label="Fetch ATPs", type="primary", on_click=process)
    state.completed_stage = start


def process() -> None:
    state.stage = process
    state.atp_connection = st.connection(
        "atpconnection", type=ATPConnection,
        compartment_id=state.compartment_id
    )
    st.subheader("Please select a database")
    cols = st.columns((4, 1, 1), )
    fields = ["ID", 'Name', 'Action']
    for col, field_name in zip(cols, fields):
        col.markdown(f":gray-background[**{field_name}**]", )
    atp_connection = cast(ATPConnection, state.atp_connection)
    for atp_id in atp_connection.atp_instances:
        atp = atp_connection.atp_instances[atp_id]
        if atp.db_name is not None:
            id, name, action = st.columns((4, 1, 1))
            id.text(body=atp.id)
            name.text(body=atp.db_name)
            params = {
                'atp_id': atp.id
            }
            action.button(key=atp.id, label="Select", on_click=select, kwargs=params)
    state.completed_stage = process


def select(**kwargs) -> None:
    state.stage = select
    if "atp_id" not in state:
        state.atp_id = kwargs.get("atp_id")
    cols = st.columns((3, 7, 2))
    fields = ["Display Name", 'Connect String', 'Action']
    for col, field_name in zip(cols, fields):
        col.markdown(f":gray-background[**{field_name}**]", )
    atp_connection = cast(ATPConnection, state.atp_connection)
    for profile in atp_connection.atp_instances[state.atp_id].connection_strings.profiles:
        if profile is not None:
            name, value, action = st.columns((3, 7, 2))
            name.text(body=profile.display_name)
            value.text(body=profile.value)
            params = {
                'connect_string': profile.value
            }
            action.button(key=profile.display_name, label="Connect", on_click=input_creds,
                          kwargs=params)
    state.completed_stage = select


def input_creds(**kwargs) -> None:
    state.stage = input_creds
    if "connect_string" not in state:
        state.connect_string = kwargs.get("connect_string")
    with st.form("input_creds_form", clear_on_submit=True, border=0):
        st.subheader("Enter Database credentials", divider="gray")
        st.text_input(label="ATP ID", value=state.atp_id, disabled=True, key='atp_id')
        st.text_input(label="Username",
                      value=os.environ.get("default_username"), key="username")
        st.text_input(label="Password", type="password", value=os.environ.get("default_password"), key="password")
        st.text_input(label="Connect string", key="connect_string")
        st.form_submit_button(label="Login", type="primary", on_click=connect)
    state.completed_stage = input_creds


def connect() -> None:
    state.stage = connect
    with st.spinner("Connecting.."):
        connection = cast(ATPConnection, state.atp_connection)
        connection.connect(
            atp_id=state.atp_id,
            user=state.username,
            creds=state.password,
            connect_string=state.connect_string)
    state.completed_stage = connect
    execute()


def execute() -> None:
    state.stage = execute
    with st.form("query_form", clear_on_submit=True, border=0):
        st.text_area(label="Enter SQL query to execute", key="query")
        col1, col2, col3 = st.columns(3)
        col1.form_submit_button(label="Execute", on_click=fetch)
        col2.form_submit_button(label="Load Data", on_click=load_data)
        col3.form_submit_button(label="Prepare Data", on_click=prepare_data)
    state.completed_stage = execute


def fetch() -> None:
    state.stage = fetch
    with st.form("result_form", clear_on_submit=True, border=0):
        with st.spinner("Loading.."):
            connection = cast(ATPConnection, state.atp_connection)
            data = connection.execute(state.query)
            st.table(data)
        col1, col2, col3 = st.columns(3)
        col1.form_submit_button(label="Execute another", on_click=execute)
        col2.form_submit_button(label="Start Again", on_click=reset)
        col3.form_submit_button(label="Load Data", on_click=load_data)
    state.completed_stage = fetch


def reset() -> None:
    state.stage = reset
    for key in st.session_state.keys():
        del st.session_state[key]
    state.completed_stage = reset


def load_data() -> None:
    state.stage = load_data
    with st.form("load_csv_form", clear_on_submit=True, border=0):
        st.subheader("Load CSV", divider="gray")
        st.text_input(label="CSV URL", key='csv_url')
        st.text_input(label="Table Name", key="table_name")
        st.checkbox(label="Drop existing table?", value=True, key="create_table")
        st.number_input(label="Char column size", value=default_char_col_size, key="char_col_size")
        col1, col2 = st.columns(2)
        col1.form_submit_button(label="Submit", type="primary", on_click=exec_load_data)
        col2.form_submit_button(label="Load another CSV", on_click=load_data)
    state.completed_stage = load_data


def exec_load_data() -> None:
    state.stage = exec_load_data
    with st.spinner("Loading Data.."):
        connection = cast(ATPConnection, state.atp_connection)
        connection.load_csv(state.table_name, state.create_table, state.char_col_size, state.csv_url, sep="\t",
                            header=0, index_col=0)
    st.button("Preview Data", on_click=connection.preview_data, args=(state.table_name, show_buttons,))
    state.completed_stage = exec_load_data


def show_buttons() -> None:
    state.stage = show_buttons
    col1, col2, col3 = st.columns(3)
    col1.button(label="Load another CSV", on_click=load_data)
    col2.button(label="Execute Query", on_click=fetch)
    col3.button(label="Prepare Data", on_click=prepare_data)
    state.completed_stage = show_buttons


def load_docs_in_parallel() -> None:
    state.stage = load_docs_in_parallel
    if "all_docs_loaded_from_db" not in state:
        with st.spinner("Loading all documents from db"):
            load_docs_from_db(False)
            state.remaining = int(len(state.docs) / 5)
            state.all_docs_loaded_from_db = True
    docs_progress_bar = st.progress(0.0, text="Updating vector store")
    with docs_progress_bar:
        step = 1 / state.remaining
        for docs_num in range(len(state.docs) - (5 * state.remaining), len(state.docs), 5):
            state.vectorstore.add_documents(chunk_data(state.docs[docs_num:docs_num + 5]))
            progress_point = (int(len(state.docs) / 5) + 1 - state.remaining) * step
            docs_loaded = docs_num + 5
            if docs_loaded >= len(state.docs):
                docs_loaded = len(state.docs)
                progress_point = 1.0
            state.remaining = state.remaining - 1
            docs_progress_bar.progress(progress_point, text=f"Loaded {docs_loaded} documents")
        state.loading_completed = True
    if state.remaining == 0:
        state.completed_stage = load_docs_in_parallel


def prepare_data() -> None:
    state.stage = prepare_data
    connection = cast(ATPConnection, state.atp_connection)
    state.onnx_model_loaded = True
    if "onnx_model_loaded" not in state:
        with st.spinner("Loading ONNX model"):
            load_onnx_model()
            state.onnx_model_loaded = True
    state.embedding_tested = True
    if "embedding_tested" not in state:
        with st.spinner("Testing embedding"):
            test_embedding()
            state.embedding_tested = True
    if "sample_docs_loaded" not in state:
        with st.spinner("Loading documents"):
            load_docs_from_db(True)
            st.markdown(f"Number of docs loaded: {len(state.docs)}")
            state.sample_docs_loaded = True
    if "vectorstore" not in state:
        with st.spinner("Init embedder"):
            embedder_params = {"provider": "database", "model": os.environ.get("model_name")}
            embedder = OracleEmbeddings(conn=connection.connection, params=embedder_params)
        with st.spinner("Init summary generator"):
            summary_params = {
                "provider": "database",
                "glevel": "S",
                "numParagraphs": 1,
                "language": "english",
            }
            state.summarizer = OracleSummary(conn=connection.connection, params=summary_params)
        with st.spinner("Init document splitter"):
            splitter_params = {"normalize": "all"}
            state.splitter = OracleTextSplitter(conn=connection.connection, params=splitter_params)
        with st.spinner("Creating vector store"):
            state.vectorstore = OracleVS.from_documents(
                chunk_data(state.docs[:5]),
                embedder,
                client=connection.connection,
                table_name="oravs",
                distance_strategy=DistanceStrategy.DOT_PRODUCT,
            )
            st.markdown(f"Vector Store Table: {state.vectorstore.table_name}")
    state.completed_stage = prepare_data
    with st.empty():
        load_docs_in_parallel()


def chunk_data(docs: list[Document]) -> list[Document]:
    chunks_with_mdata = []
    for id, doc in enumerate(docs, start=1):
        summ = state.summarizer.get_summary(doc.page_content)
        chunks = state.splitter.split_text(doc.page_content)
        for ic, chunk in enumerate(chunks, start=1):
            chunk_metadata = doc.metadata.copy()
            #chunk_metadata["id"] = str(chunk_metadata["product_id"]) + "$" + str(id) + "$" + str(ic)
            chunk_metadata["product_text_id"] = str(id)
            chunk_metadata["product_text_summary"] = str(summ[0])
            chunks_with_mdata.append(
                Document(page_content=str(chunk), metadata=chunk_metadata)
            )
    return chunks_with_mdata


def load_docs_from_db(limit: bool) -> None:
    connection = cast(ATPConnection, state.atp_connection)
    sql_stmt = "select product_id, product_name, COALESCE(product_description, product_name) as product_text from products"
    if limit:
        sql_stmt = f"{sql_stmt} FETCH FIRST 200 ROWS ONLY"
    product_text_pd = pd.read_sql(
        sql_stmt,
        connection.engine)
    state.docs = (
        DataFrameLoader(
            product_text_pd,
            page_content_column='product_text'
        )
        .load()
    )


def load_onnx_model() -> None:
    state.stage = load_onnx_model
    connection = cast(ATPConnection, state.atp_connection)
    try:
        if connection.connection is None or os.environ.get("model_dir") is None or os.environ.get(
                "model_file") is None or os.environ.get("model_name") is None:
            raise Exception("Invalid input")
        with open(f"{os.environ.get("model_dir")}/{os.environ.get("model_file")}", 'rb') as f:
            model_data = f.read()
        curr = connection.connection.cursor()
        curr.execute(
            """
            begin
                dbms_data_mining.drop_model(model_name => :model_name, force => true);
                SYS.DBMS_VECTOR.load_onnx_model(:model_name, :model_data, 
                    json('{"function" : "embedding", 
                        "embeddingOutput" : "embedding", 
                        "input": {"input": ["DATA"]}}'));
            end;""",
            model_name=os.environ.get("model_name"),
            model_data=model_data
        )
        curr.close()

    except Exception as ex:
        curr.close()
        raise
    state.completed_stage = load_onnx_model


def test_embedding() -> None:
    state.stage = test_embedding
    connection = cast(ATPConnection, state.atp_connection)
    embedder_params = {"provider": "database", "model": os.environ.get("model_name")}
    embedder = OracleEmbeddings(conn=connection.connection, params=embedder_params)
    embed = embedder.embed_query("Hello World!")
    st.markdown(f"Embedding for Hello World! generated by OracleEmbeddings: {embed}")
    state.completed_stage = test_embedding


if "stage" in state:
    print(state.stage)
if ("stage" in state) and (
        ("completed_stage" not in state) or (state.completed_stage.__name__ != state.stage.__name__)):
    print("starting " + state.stage.__name__)
    state.stage()
else:
    if "stage" not in state:
        collect_data()
