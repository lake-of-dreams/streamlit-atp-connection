"""Streamlit connection to an Oracle Autonomous DB."""
import base64
import numbers
from os import PathLike
from typing import Any, Dict, Iterator, Iterable, Union, Literal, Optional, cast, Sequence, Hashable
from datetime import timedelta

import oracledb
import pandas as pd
from pandas import DataFrame
from pandas._typing import ReadCsvBuffer
from streamlit.connections import BaseConnection

import streamlit as st
import oci
from .utils import oci_config, extract_zip, write_files, random_passwd
import io
import sqlalchemy as sa

class ATPConnection(BaseConnection):
    config = dict(oci.config.DEFAULT_CONFIG)
    atp_instances : Dict[str, oci.database.models.autonomous_database.AutonomousDatabase] = dict()
    connection : oracledb.Connection = None
    db_client: oci.database.DatabaseClient = None
    engine: sa.engine.Engine = None

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
            self.engine = sa.create_engine('oracle+oracledb://', creator=lambda: self.connection)

    def execute(self, query: str) -> Any:
        return pd.read_sql(query, self.engine)

    def load_csv(self, table_name: str, create_table: bool, char_column_size: numbers.Number, filepath_or_buffer: str | PathLike[str] | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], sep: str | None, header: int | Sequence[int] | None | Literal["infer"] = "infer", index_col: Hashable | Sequence[Hashable] | Literal[False] | None = None, ) -> DataFrame | Iterator[DataFrame]:
        df = pd.read_csv(filepath_or_buffer, sep=sep, header=header,
                         index_col=index_col)
        cursor = self.connection.cursor()

        if create_table:
            schema = pd.io.json.build_table_schema(df)
            cols_part = ""
            pk_part = ""
            for item in schema.items():
                match item[0]:
                    case "fields":
                        for field in item[1]:
                            cols_part = f"{cols_part} {field['name']} "
                            match field['type']:
                                case "integer":
                                    cols_part = f"{cols_part} NUMBER, "
                                case "string":
                                    cols_part = f"{cols_part} VARCHAR2({char_column_size}), "
                    case "primaryKey":
                        for primary_key in item[1]:
                            pk_part = f"{primary_key},{pk_part}"
            drop_stmt = f"DROP TABLE IF EXISTS {table_name} cascade constraints purge"
            create_stmt = f"CREATE TABLE {table_name} ({cols_part} PRIMARY KEY ({pk_part[:-1]}))"
            cursor.execute(drop_stmt)
            cursor.execute(create_stmt)
        df.to_sql(name=table_name,con=self.engine,if_exists="replace",index=True)



    def preview_data(self, table_name: str) -> None:
        df = pd.read_sql(f"select * from {table_name} FETCH FIRST 50 ROWS ONLY", self.engine, chunksize=10)
        st.table(df)




