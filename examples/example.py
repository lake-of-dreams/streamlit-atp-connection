import os

import streamlit as st

from atp_connection import ATPConnection
from typing import Any,  Dict, Iterator, Iterable, Union, Literal, Optional, cast
from dotenv import load_dotenv

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
        st.form_submit_button(label="Execute", on_click=fetch)
        st.form_submit_button(label="Load Data", on_click=load_data)


def fetch() -> None:
    with st.form("result_form", clear_on_submit=True, border=0):
        connection = cast(ATPConnection, state.atp_connection)
        data = connection.execute(state.query)
        st.table(data)
        st.form_submit_button(label="Execute another", on_click=execute)
        st.form_submit_button(label="Start Again", on_click=reset)
        st.form_submit_button(label="Load Data", on_click=load_data)

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
        st.form_submit_button(label="Submit", type="primary", on_click=exec_load_data)
        st.form_submit_button(label="Load another CSV", on_click=load_data)

def exec_load_data() -> None:
    with st.form("exec_load_csv_form", clear_on_submit=True, border=0):
        connection = cast(ATPConnection, state.atp_connection)
        connection.load_csv(state.table_name, state.create_table, state.char_col_size, state.csv_url,sep="\t",header=0,index_col=0)
        st.form_submit_button("Preview Data", on_click=connection.preview_data, args=(state.table_name,))
        st.form_submit_button(label="Load another CSV", on_click=load_data)
        st.form_submit_button(label="Execute Query", on_click=fetch)

run_example()

