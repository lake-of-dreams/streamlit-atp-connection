""" Streamlit ATP connection example."""
import os
from enum import StrEnum
from typing import cast

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import OracleTextSplitter
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.utilities import OracleSummary
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from atp_connection import ATPConnection
from atp_connection.utils import write_files

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

class Choices(StrEnum):
    use_default_config = "Use Default Config ($HOME/.oci/config)"
    select_config = "Select a config file"
    enter_config = "Enter OCI config details"
    upload_config = "Upload config file"
    enter_config_location = "Enter config file location"
    upload_key_file = "Upload Key file"
    enter_key_file_location = "Enter key file location"
    oci_profile_default = "DEFAULT"
    oci_profile_enter = "Enter OCI Profile"
    upload_model = "Upload model file"
    enter_model_location = "Enter Model file location"


def collect_oci_config() -> None:
    state.stage = collect_oci_config
    state.form_error = None
    state.oci_profile = "DEFAULT"
    state.oci_passphrase = ""
    if st.radio("Specify OCI config:",
             [Choices.use_default_config, Choices.select_config, Choices.enter_config], key="specify_config",
             ):
        match state.specify_config:
            case Choices.select_config:
                if st.radio("", [Choices.upload_config, Choices.enter_config_location],
                         key="specify_select_config"):
                    match state.specify_select_config:
                        case Choices.upload_config:
                            st.file_uploader("Choose oci config file", None, False,
                                             key="uploaded_config_file", )
                        case Choices.enter_config_location:
                            st.text_input("OCI config file location",
                                          key="config_file_location")
            case Choices.enter_config:
                st.text_input("User OCID:", key="oci_user", )
                st.text_input("Key Fingerprint:", key="oci_fingerprint", )
                st.text_input("Key Passphrase:", key="oci_passphrase", type="password")
                st.text_input("Tenancy OCID:", key="oci_tenancy")
                st.text_input("Region:", key="oci_region", )
                if st.radio("", [Choices.upload_key_file, Choices.enter_key_file_location], key="specify_key_file"):
                    match state.specify_key_file:
                        case Choices.upload_key_file:
                            st.file_uploader("Choose oci key file", None, False,
                                             key="uploaded_key_file")
                        case Choices.enter_key_file_location:
                            st.text_input("OCI key file location", None, False,
                                          key="oci_keyfile")

        if (state.specify_config == Choices.use_default_config) or (state.specify_config ==  Choices.select_config):
            if st.radio("OCI Config profile",
                        [Choices.oci_profile_default, Choices.oci_profile_enter],
                        key="specify_oci_profile"):
                match state.specify_oci_profile:
                    case Choices.oci_profile_default:
                        state.oci_profile = Choices.oci_profile_default
                    case Choices.oci_profile_enter:
                        st.text_input("OCI config profile", key="oci_profile")
    st.button(label="Connect", type="primary", on_click=create_connection)

def create_connection() -> None:
    match state.specify_config:
        case Choices.use_default_config:
            state.atp_connection = st.connection(
                "atpconnection", type=ATPConnection,
                oci_profile=state.oci_profile
            )
        case Choices.select_config:
            match state.specify_select_config:
                case Choices.upload_config:
                    if state.uploaded_config_file is not None:
                        state.config_file_location = write_files(state.uploaded_config_file.getvalue(), False)
                    else:
                        state.form_error = st.error("OCI config file not uploaded")
            if (state.config_file_location is not None) and (state.config_file_location != ""):
                state.atp_connection = st.connection(
                    "atpconnection", type=ATPConnection,
                    config_file=state.config_file_location,
                    oci_profile=state.oci_profile
                )
            else:
                state.form_error = st.error("OCI config file location not specified")
        case Choices.enter_config:
            match state.specify_key_file:
                case Choices.upload_key_file:
                    if state.uploaded_key_file is not None:
                        state.oci_keyfile = write_files(state.uploaded_key_file.getvalue(), False)
                    else:
                        form_error = st.error("OCI key file not uploaded")
            if (state.oci_keyfile is None) or (state.oci_keyfile == ""):
                state.form_error = st.error("OCI key file location not specified")
            if (state.oci_user is None) or (state.oci_user == ""):
                state.form_error = st.error("OCI User OCID not specified")
            if (state.oci_fingerprint is None) or (state.oci_fingerprint == ""):
                state.form_error = st.error("Key fingerprint not specified")
            if (state.oci_tenancy is None) or (state.oci_tenancy == ""):
                state.form_error = st.error("Tenancy not specified")
            if (state.oci_region is None) or (state.oci_region == ""):
                state.form_error = st.error("Region not specified")
            if state.form_error is None:
                config = {
                    "oci_keyfile": state.oci_keyfile,
                    "oci_user": state.oci_user,
                    "oci_fingerprint": state.oci_fingerprint,
                    "oci_tenancy": state.oci_tenancy,
                    "oci_region": state.oci_region,
                    "oci_passphrase": state.oci_passphrase
                }
                state.atp_connection = st.connection(
                    "atpconnection", type=ATPConnection,
                    config=config
                )

    if state.form_error is not None:
        st.markdown("Please fix errors :error")
    else:
        state.completed_stage = collect_oci_config
        state.stage = start


def start() -> None:
    state.stage = start
    atp_connection = cast(ATPConnection, state.atp_connection)
    st.text_input(label="Compartment ID", value=os.environ.get("default_compartment_id"), key="compartment_id")
    if st.button(label="Fetch ATPs", type="primary"):
        atp_connection.fetch_atp_instances(state.compartment_id)
        process()


def process() -> None:
    state.completed_stage = start
    state.stage = process
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


def select(**kwargs) -> None:
    state.completed_stage = process
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
    state.stage = execute


def execute() -> None:
    state.stage = execute
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.button(label="Execute SQL Query", on_click=run_query)
    col2.button(label="Load CSV Data",on_click=load_data)
    col3.button(label="Load ONNX Model", on_click=load_onnx_model)
    col4.button(label="Generate Embeddings", on_click=test_embedding)
    col5.button(label="Prepare Products Data", on_click=prepare_data)
    state.completed_stage = execute

def run_query() -> None:
    state.stage = run_query
    with st.form("sql_query", clear_on_submit=True, enter_to_submit=False, border=0):
        st.text_area(label="Enter SQL query to execute", key="query")
        col1, col2 = st.columns(2)
        col1.form_submit_button("Run", on_click=fetch)
        col2.form_submit_button("Back", on_click=execute)
        state.completed_stage = run_query


def fetch() -> None:
    state.stage = fetch
    with st.spinner("Loading.."):
        connection = cast(ATPConnection, state.atp_connection)
        data = connection.execute(state.query)
        st.table(data)
        st.button("Back", on_click=run_query)
        state.completed_stage = fetch


def reset() -> None:
    state.stage = reset
    for key in st.session_state.keys():
        del st.session_state[key]
    state.completed_stage = reset


def load_data() -> None:
    state.stage = load_data
    with st.form("load_data", clear_on_submit=True, enter_to_submit=False, border=0):
        st.subheader("Load CSV", divider="gray")
        st.text_input(label="CSV URL", key='csv_url')
        st.text_input(label="Table Name", key="table_name")
        st.checkbox(label="Drop existing table?", value=True, key="create_table")
        st.number_input(label="Char column size", value=default_char_col_size, key="char_col_size")
        col1, col2 = st.columns(2)
        col1.form_submit_button("Load", on_click=conn_load_csv)
        col2.form_submit_button("Back", on_click=execute)
        state.completed_stage = load_data

def conn_load_csv() -> None:
    state.stage = conn_load_csv
    with st.spinner("Loading.."):
        connection = cast(ATPConnection, state.atp_connection)
        connection.load_csv(state.table_name, state.create_table, state.char_col_size, state.csv_url, sep="\t",
                            header=0, index_col=0)
        sample_data = connection.sample_data(state.table_name)
        st.subheader("Preview Data")
        st.table(sample_data)
        st.button("Back", on_click=load_data)
        state.completed_stage = conn_load_csv

def load_docs_in_parallel() -> None:
    state.stage = load_docs_in_parallel
    st.button("Back", on_click=execute)
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
    # Example assumes that model from https://oracle-base.com/articles/23/ai-vector-search-23 is loaded with name all_MiniLM_L12_v2
    if "model_name" not in state:
        state.model_name = "all_MiniLM_L12_v2"
    if "sample_docs_loaded" not in state:
        with st.spinner("Loading documents"):
            load_docs_from_db(True)
            st.markdown(f"Number of docs loaded: {len(state.docs)}")
            state.sample_docs_loaded = True
    if "vectorstore" not in state:
        with st.spinner("Init embedder"):
            embedder_params = {"provider": "database", "model": state.model_name}
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
                chunk_data(state.docs),
                embedder,
                client=connection.connection,
                table_name="oravs",
                distance_strategy=DistanceStrategy.DOT_PRODUCT,
            )
            st.markdown(f"Vector Store Table: {state.vectorstore.table_name}")
    state.completed_stage = prepare_data
    st.button("Start over", key="back_to_execute",on_click=execute)
    with st.empty():
        load_docs_in_parallel()


def chunk_data(docs: list[Document]) -> list[Document]:
    chunks_with_mdata = []
    for id, doc in enumerate(docs, start=1):
        summ = state.summarizer.get_summary(doc.page_content)
        chunks = state.splitter.split_text(doc.page_content)
        for ic, chunk in enumerate(chunks, start=1):
            chunk_metadata = doc.metadata.copy()
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
        sql_stmt = f"{sql_stmt} FETCH FIRST 50 ROWS ONLY"
    state.docs = connection.load_records_as_documents(sql_stmt, "product_text")

def load_onnx_model() -> None:
    state.stage = load_onnx_model
    with st.form("load_onnx_model", clear_on_submit=True, enter_to_submit=False, border=0):
        st.subheader("Load ONNX Model", divider="gray")
        state.model_location = os.environ.get("model_file") or ""
        state.model_name = os.environ.get("model_name") or ""
        if (state.model_location is None) or (state.model_location == "") or (state.model_name is None) or (
                state.model_name == ""):
            st.text_input("Enter model name", key="model_name")
            st.file_uploader("Choose model file", None, False,
                             key="uploaded_model_file", )
            st.text_input("Model file location",
                          key="model_location", disabled=(state.uploaded_model_file is not None))
        col1, col2 = st.columns(2)
        col1.form_submit_button("Load", on_click=conn_load_onnx_model)
        col2.form_submit_button("Back", on_click=execute)
        state.completed_stage = load_onnx_model

def conn_load_onnx_model() -> None:
    state.stage = conn_load_onnx_model
    with st.spinner("Loading.."):
        connection = cast(ATPConnection, state.atp_connection)
        if state.uploaded_model_file is not None:
            state.model_location = write_files(state.uploaded_model_file.getvalue(), False)
        connection.load_onnx_model(state.model_location, state.model_name)
        st.markdown("Successfully loaded model")
        st.button("Back", on_click=load_onnx_model)
        state.completed_stage = conn_load_onnx_model

def test_embedding() -> None:
    state.stage = test_embedding
    with st.form("test_embedding", clear_on_submit=True, enter_to_submit=False, border=0):
        st.subheader("Generate Embedding", divider="gray")
        st.text_input("Enter text to generate embedding", key="embed_query")
        if "model_name" not in state:
            st.text_input("Enter model name to generate embedding", key="model_name")
        col1, col2 = st.columns(2)
        col1.form_submit_button("Go", on_click=conn_test_embedding)
        col2.form_submit_button("Back", on_click=execute)
        state.completed_stage = test_embedding

def conn_test_embedding() -> None:
    state.stage = conn_test_embedding
    with st.spinner("Loading.."):
        connection = cast(ATPConnection, state.atp_connection)
        st.markdown(
            f"Embedding for {state.embed_query} generated by model {state.model_name}: {connection.create_embedding(state.model_name, state.embed_query)}")
        st.button("Back", on_click=test_embedding)
        state.completed_stage = conn_test_embedding

if ("stage" in state) and (
        ("completed_stage" not in state) or (state.completed_stage.__name__ != state.stage.__name__)):
    state.stage()
else:
    if "stage" not in state:
        collect_oci_config()
