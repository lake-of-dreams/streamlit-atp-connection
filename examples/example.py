import os
from typing import cast

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from streamlit.delta_generator import DeltaGenerator
from streamlit_extras.grid import GridDeltaGenerator
from streamlit_extras.row import row

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


def start() -> None:
    state.started = True
    with st.form("compartment_id_form", clear_on_submit=True, border=0):
        st.text_input(label="Compartment ID", value=os.environ.get("default_compartment_id"), key="compartment_id")
        st.form_submit_button(label="Fetch ATPs", type="primary", on_click=process)


def process() -> None:
    state.atp_connection = st.connection(
        "atpconnection", type=ATPConnection,
        compartment_id=state.compartment_id
    )
    show()


def show() -> None:
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
            action.button(key=atp.id, label="Select", on_click=select, args=(atp.id,))


def select(atp_id: str) -> None:
    state.atp_id = atp_id
    cols = st.columns((3, 7, 2))
    fields = ["Display Name", 'Connect String', 'Action']
    for col, field_name in zip(cols, fields):
        col.markdown(f":gray-background[**{field_name}**]", )
    atp_connection = cast(ATPConnection, state.atp_connection)
    for profile in atp_connection.atp_instances[atp_id].connection_strings.profiles:
        if profile is not None:
            name, value, action = st.columns((3, 7, 2))
            name.text(body=profile.display_name)
            value.text(body=profile.value)
            action.button(key=profile.display_name, label="Connect", on_click=input_creds,
                          args=(profile.value,))


def input_creds(connect_string: str) -> None:
    with st.form("input_creds_form", clear_on_submit=True, border=0):
        st.subheader("Enter Database credentials", divider="gray")
        st.text_input(label="ATP ID", value=state.atp_id, disabled=True, key='atp_id')
        st.text_input(label="Username",
                      value=os.environ.get("default_username"), key="username")
        st.text_input(label="Password", type="password", value=os.environ.get("default_password"), key="password")
        st.text_input(label="Connect string", value=connect_string, key="connect_string")
        st.form_submit_button(label="Login", type="primary", on_click=connect)


def connect() -> None:
    with st.spinner("Connecting.."):
        connection = cast(ATPConnection, state.atp_connection)
        connection.connect(
            atp_id=state.atp_id,
            user=state.username,
            creds=state.password,
            connect_string=state.connect_string)
    execute()


def execute() -> None:
    with st.form("query_form", clear_on_submit=True, border=0):
        st.text_area(label="Enter SQL query to execute", key="query")
        col1, col2, col3 = st.columns(3)
        col1.form_submit_button(label="Execute", on_click=fetch)
        col2.form_submit_button(label="Load Data", on_click=load_data)
        col3.form_submit_button(label="Run Vector search Demo", on_click=run_vector_search_demo)


def fetch() -> None:
    with st.form("result_form", clear_on_submit=True, border=0):
        with st.spinner("Loading.."):
            connection = cast(ATPConnection, state.atp_connection)
            data = connection.execute(state.query)
            st.table(data)
        col1, col2, col3 = st.columns(3)
        col1.form_submit_button(label="Execute another", on_click=execute)
        col2.form_submit_button(label="Start Again", on_click=reset)
        col3.form_submit_button(label="Load Data", on_click=load_data)


def reset() -> None:
    for key in st.session_state.keys():
        del st.session_state[key]


def run_example() -> None:
    if "started" not in state or not state.started:
        with st.form("main_page", clear_on_submit=True, border=0):
            st.form_submit_button("Connect to an ATP", on_click=start)


def load_data() -> None:
    with st.form("load_csv_form", clear_on_submit=True, border=0):
        st.subheader("Load CSV", divider="gray")
        st.text_input(label="CSV URL", key='csv_url')
        st.text_input(label="Table Name", key="table_name")
        st.checkbox(label="Drop existing table?", value=True, key="create_table")
        st.number_input(label="Char column size", value=default_char_col_size, key="char_col_size")
        col1, col2 = st.columns(2)
        col1.form_submit_button(label="Submit", type="primary", on_click=exec_load_data)
        col2.form_submit_button(label="Load another CSV", on_click=load_data)


def exec_load_data() -> None:
    with st.spinner("Loading Data.."):
        connection = cast(ATPConnection, state.atp_connection)
        connection.load_csv(state.table_name, state.create_table, state.char_col_size, state.csv_url, sep="\t",
                            header=0, index_col=0)
    st.button("Preview Data", on_click=connection.preview_data, args=(state.table_name, show_buttons,))
    show_buttons()


def show_buttons() -> None:
    col1, col2, col3 = st.columns(3)
    col1.button(label="Load another CSV", on_click=load_data)
    col2.button(label="Execute Query", on_click=fetch)
    col3.button(label="Run Vector search Demo", on_click=run_vector_search_demo)


def run_vector_search_demo() -> None:
    connection = cast(ATPConnection, state.atp_connection)
    print(os.environ.get("model_dir"))
    print(os.environ.get("model_file"))
    print(os.environ.get("model_name"))
    # with st.spinner("Loading ONNX model"):
    # load_onnx_model()
    # with st.spinner("Testing embedding"):
    # test_embedding()
    with st.spinner("Loading documents"):
        load_docs_from_db(True)
        st.markdown(f"Number of docs loaded: {len(state.docs)}")
    with st.spinner("Init embedder"):
        embedder_params = {"provider": "database", "model": os.environ.get("model_name")}
        embedder = OracleEmbeddings(conn=connection.connection, params=embedder_params)
    with st.spinner("Creating vector store"):
        state.vectorstore = OracleVS.from_documents(
            state.docs[:5],
            embedder,
            client=connection.connection,
            table_name="oravs",
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
        st.markdown(f"Vector Store Table: {state.vectorstore.table_name}")
    col1, col2 = st.columns([1,1])
    with col1:
        with st.popover("Search"):
            search()

    with col2:
        with st.popover("Loading status"):
            with st.spinner("Loading all documents from db"):
                load_docs_from_db(False)
            docs_progress_bar = st.progress(0.0, text="Updating vector store")
            with docs_progress_bar:
                remaining = len(state.docs) / 5
                step = 1 / remaining
                counter = 1
                for docs_num in range(0, len(state.docs), 5):
                    state.vectorstore.add_documents(state.docs[docs_num:docs_num + 5])
                    progress_point = counter * step
                    docs_loaded = docs_num + 5
                    if docs_loaded >= len(state.docs):
                        docs_loaded = len(state.docs)
                        progress_point = 1.0
                    counter = counter + 1
                    docs_progress_bar.progress(progress_point, text=f"Loaded {docs_loaded} documents")
                state.loading_completed = True


@st.dialog("Search", width="large")
def search():
    if 'go_clicked' not in state:
        state.go_clicked = False

    def click_button():
        state.go_clicked = True

    search_query = st.text_input(label="Search", value="red carpet")
    st.button(label="Go", on_click=click_button)

    if state.go_clicked:
        state.search_results = state.vectorstore.similarity_search(search_query)
        st.write(state.search_results)


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


def test_embedding() -> None:
    connection = cast(ATPConnection, state.atp_connection)
    embedder_params = {"provider": "database", "model": os.environ.get("model_name")}
    embedder = OracleEmbeddings(conn=connection.connection, params=embedder_params)
    embed = embedder.embed_query("Hello World!")
    st.markdown(f"Embedding for Hello World! generated by OracleEmbeddings: {embed}")


run_example()
