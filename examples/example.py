import streamlit as st

from atp_connection import ATPConnection

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
</style>""", unsafe_allow_html=True)


def start() -> None:
    val_compartment_id = st.text_input(label="Compartment ID",
                                       value="")
    st.button(label="Fetch ATPs", type="primary", on_click=process, args=(val_compartment_id,))

def process(compartment_id: str) -> None:
    conn = st.connection(
        "atpconnection", type=ATPConnection,
        compartment_id=compartment_id
    )
    show(conn)


def show(connection: ATPConnection) -> None:
    st.subheader("Please select a database")
    cols = st.columns((4, 1, 1), )
    fields = ["ID", 'Name', 'Action']
    for col, field_name in zip(cols, fields):
        col.markdown(f":gray-background[**{field_name}**]", )
    for atp_id in connection.atp_instances:
        atp = connection.atp_instances[atp_id]
        if atp.db_name is not None:
            id, name, action = st.columns((4, 1, 1))
            id.text(body=atp.id)
            name.text(body=atp.db_name)
            action.button(key=atp.id, label="Select", on_click=select, args=(connection, atp.id))


def select(connection: ATPConnection, atp_id: str) -> None:
    cols = st.columns((3, 7, 2))
    fields = ["Display Name", 'Connect String', 'Action']
    for col, field_name in zip(cols, fields):
        col.markdown(f":gray-background[**{field_name}**]", )
    for profile in connection.atp_instances[atp_id].connection_strings.profiles:
        if profile is not None:
            name, value, action = st.columns((3, 7, 2))
            name.text(body=profile.display_name)
            value.text(body=profile.value)
            action.button(key=profile.display_name, label="Connect", on_click=input_creds,
                          args=(connection, profile.value, atp_id))


def input_creds(connection: ATPConnection, connect_string: str, atp_id: str) -> None:
    st.subheader("Enter Database credentials", divider="gray")
    st.text_input(label="ATP ID", value=atp_id, disabled=True)
    user = st.text_input(label="Username",
                         value="ADMIN")
    passwd = st.text_input(label="Password", type="password", value="")
    val_connect_string = st.text_input(label="Connect string", value=connect_string)
    st.button(key="login", label="Login", on_click=connect, args=(connection, atp_id, val_connect_string, user, passwd))


def connect(connection: ATPConnection, atp_id: str, connect_string: str, user: str, creds: str) -> None:
    connection.connect(
        atp_id=atp_id,
        user=user,
        creds=creds,
        connect_string=connect_string)
    show_query_input(connection)


def show_query_input(connection: ATPConnection) -> None:
    query = st.text_area(label="Enter SQL query to execute", value="select 1 from dual")
    st.button(key="execute", label="Execute", on_click=execute, args=(connection, query))


def execute(connection: ATPConnection, query: str) -> None:
    cursor = connection.execute(query)
    for rows in cursor:
        st.dataframe(rows)
    st.button(key="ExecuteQuery", label="Execute Query", on_click=show_query_input, args=(connection,))

st.button("Connect to an ATP", on_click=start, args=(), )