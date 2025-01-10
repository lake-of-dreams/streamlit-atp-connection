"""Streamlit connection to an Oracle Autonomous DB."""
import base64
from typing import Any,  Dict, Iterator, Iterable, Union, Literal, Optional, cast
from datetime import timedelta

import oracledb
from streamlit.connections import BaseConnection

import streamlit as st
import oci
from .utils import oci_config, extract_zip, write_files, random_passwd
import io

class ATPConnection(BaseConnection):
    config = dict(oci.config.DEFAULT_CONFIG)
    atp_instances : Dict[str, oci.database.models.autonomous_database.AutonomousDatabase] = dict()
    connection : oracledb.Connection = None
    db_client: oci.database.DatabaseClient = None

    def __init__(self,
        connection_name: str, **kwargs
    ) -> None:
        input_config = kwargs.get("config")
        self.config = oci_config(input_config)
        super().__init__(connection_name, **kwargs)

    def _connect(self, **kwargs) -> None:
        self.db_client = oci.database.DatabaseClient(self.config)
        response = self.db_client.list_autonomous_databases(kwargs.get("compartment_id"))
        for atp_data in response.data:
            atp = cast(oci.database.models.autonomous_database.AutonomousDatabase, atp_data)
            self.atp_instances[atp.id] = atp

    def connect(self, atp_id:str, connect_string: str, user: str, creds: str) -> None:
        wallet_passwd = random_passwd()
        wallet_options = oci.database.models.GenerateAutonomousDatabaseWalletDetails(generate_type="ALL",password=wallet_passwd,is_regional=False)
        wallet_response = self.db_client.generate_autonomous_database_wallet(autonomous_database_id=atp_id, generate_autonomous_database_wallet_details=wallet_options)
        mem_file = io.BytesIO(wallet_response.data.content)
        wallet_location = write_files(mem_file)
        ora_connection = oracledb.connect(
            user=user,
            password=creds,
            dsn=connect_string,
            wallet_location=wallet_location,
            wallet_password=wallet_passwd)
        if ora_connection.is_healthy():
            st.markdown("Successfully connected to Oracle Database")
            self.connection = ora_connection

    def execute(self, query: str) -> oracledb.Cursor:
        return self.connection.cursor().execute(query)

