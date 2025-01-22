"""Streamlit connection for an Oracle Autonomous DB and associated utilities."""
import io
import numbers
from os import PathLike
from typing import Any, Dict, Iterator, Literal, cast, Sequence, Hashable

import oci
import oracledb
import pandas as pd
import sqlalchemy as sa
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import OracleEmbeddings
from langchain_core.documents import Document
from oci.database.models import AutonomousDatabase
from oracledb import Connection
from pandas import DataFrame
from streamlit.connections import BaseConnection

from .utils import oci_config, write_files, random_passwd


class ATPConnection(BaseConnection):
    """Connects a streamlit app to an Oracle Autonomous Database in Oracle Cloud Infrastructure.
    Provides utility functions that can be used to access AI features of the Oracle Database.
    """
    config = dict(oci.config.DEFAULT_CONFIG)
    atp_instances : Dict[str, oci.database.models.autonomous_database.AutonomousDatabase] = dict()
    connection : oracledb.Connection = None
    db_client: oci.database.DatabaseClient = None
    engine: sa.engine.Engine = None

    def __init__(self,
        connection_name: str, **kwargs
    ) -> None:
        """
        :param connection_name: the name of the connection

        :param kwargs:

        Additionally, the following optional keyword arguments can be used to configure the underlying OCI client:
            1. Specify OCI access parameters
           - `oci_cli_region` or `oci_region` (str) - OCI Region name
           - `oci_cli_user or` `oci_user` (str) - OCI username
           - `oci_cli_fingerprint` or `oci_fingerprint` (str) - OCI Access Key Fingerprint
           - `oci_cli_keyfile` or `oci_keyfile` (str) - OCI Access Key File location
           - `optional oci_cli_passphrase` or `oci_passphrase` (str) - OCI Access Key passphrase
           - `oci_cli_tenancy or` `oci_tenancy` (str) - OCI User tenancy
            note::
            OCI access parameters can also be specified in the streamlit secrets file (.streamlit/secrets.toml)
           Or

           2. `config` (str) with value as a dict containing OCI access parameters

           Or

           3. `config_file` (str) with value as location of OCI config file(https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm)

           In absence of these three arguments, the OCI client will be created with default OCI config from `~/.oci/config`.
           Also, `oci_profile` argument can be passed to specify the profile to use from config file, default being DEFAULT.
        """
        if self._secrets is not None:
            config_from_secrets = self._secrets.to_dict()
            kwargs.update(config_from_secrets)
        self.config = oci_config(**kwargs)
        super().__init__(connection_name, **kwargs)

    def _connect(self, **kwargs) -> None:
        self.db_client = oci.database.DatabaseClient(self.config)

    def fetch_atp_instances(self, compartment_id: str) -> dict[str, AutonomousDatabase]:
        """Accesses Autonomous Database instances for a given compartment.
        Parameters
        ----------
        compartment_id : str
            The ``OCID`` of the compartment.
        Returns
        -------
        dict
            dict containing key as ``OCID`` of the Autonomous Database instance with value as corresponding ``AutonomousDatabase`` object.
        """
        response = self.db_client.list_autonomous_databases(compartment_id)
        for atp_data in response.data:
            atp = cast(oci.database.models.autonomous_database.AutonomousDatabase, atp_data)
            self.atp_instances[atp.id] = atp
        return self.atp_instances

    def connect(self, atp_id:str, connect_string: str, user: str, creds: str) -> Connection:
        """
        Connects to an Autonomous database instance and downloads its wallet and creates a connection.
        :param atp_id: OCID of Autonomous database instance
        :param connect_string: The connect string used for the connection
        :param user: DB Username
        :param creds: DDB Password
        :return: connection object
        """
        wallet_passwd = random_passwd()
        wallet_options = oci.database.models.GenerateAutonomousDatabaseWalletDetails(generate_type="ALL",password=wallet_passwd,is_regional=False)
        wallet_response = self.db_client.generate_autonomous_database_wallet(autonomous_database_id=atp_id, generate_autonomous_database_wallet_details=wallet_options)
        mem_file = io.BytesIO(wallet_response.data.content)
        wallet_location = write_files(mem_file, True)
        ora_connection = oracledb.connect(
            user=user,
            password=creds,
            dsn=connect_string,
            wallet_location=wallet_location,
            wallet_password=wallet_passwd)
        if ora_connection.is_healthy():
            self.connection = ora_connection
            self.engine = sa.create_engine('oracle+oracledb://', creator=lambda: self.connection)
        return self.connection

    def execute(self, query: str) -> Any:
        """
        Executes a query against the DB and returns the result.
        :param query: SQL query to execute
        :return: results
        """
        return pd.read_sql(query, self.engine)

    def load_csv(self, table_name: str, create_table: bool, char_column_size: numbers.Number, filepath_or_buffer: str | PathLike[str] , sep: str | None, header: int | Sequence[int] | None | Literal["infer"] = "infer", index_col: Hashable | Sequence[Hashable] | Literal[False] | None = None, ) -> DataFrame | Iterator[DataFrame]:
        """
        Loads a CSV into a Database table
        :param table_name: Name of the table
        :param create_table: Flag indicating whether to create a new table, dropping any existing table of same name.
        :param char_column_size: Character size of Varchar column
        :param filepath_or_buffer: Location of CSV
        :param sep: CSV separator
        :param header: Header row
        :param index_col: Index column
        :return:
        """
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
                            col_name = field['name'].replace(' ','_')
                            cols_part = f"{cols_part} {col_name} "
                            match field['type']:
                                case "number":
                                    cols_part = f"{cols_part} NUMBER, "
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
        dtyp = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                dtyp[column] = sa.types.VARCHAR(df[column].astype(str).str.len().max())
            elif df[column].dtype in ['float', 'float64']:
                dtyp[column] = sa.FLOAT
        df.to_sql(name=table_name,con=self.engine,if_exists="replace",index=True,dtype=dtyp)

    def sample_data(self, table_name: str) -> DataFrame:
        """
        Samples first few records from a table.
        :param table_name: Name of the table
        :return: sample rows from the table.
        """
        df = pd.DataFrame()
        reader = pd.read_sql(f"select * from {table_name} FETCH FIRST 50 ROWS ONLY",
                             self.engine,
                             chunksize=10)
        for chunk in reader:
            df = pd.concat([df, chunk], ignore_index=True)
        return df

    def load_onnx_model(self, model_location: str, model_name: str) -> None:
        """
        Loads an ONNX model from a file into the database.
        :param model_location: Location of model file.
        :param model_name: Name of the model
        :return:
        """
        try:
            if (model_location is None) or (model_name is None):
                raise Exception("Invalid input")
            with open(model_location, 'rb') as f:
                model_data = f.read()
            curr = self.connection.cursor()
            curr.execute(
                """
                begin
                    dbms_data_mining.drop_model(model_name => :model_name, force => true);
                    SYS.DBMS_VECTOR.load_onnx_model(:model_name, :model_data, 
                        json('{"function" : "embedding", 
                            "embeddingOutput" : "embedding", 
                            "input": {"input": ["DATA"]}}'));
                end;""",
                model_name=model_name,
                model_data=model_data
            )
            curr.close()
        except Exception as ex:
            curr.close()
            raise

    def create_embedding(self, model_name: str, query: str) -> list[float]:
        """
        Creates embeddings from a loaded model in database for the input query.
        :param model_name: Name of the model already loaded in database
        :param query: Phrase to create embedding
        :return: Generated embedding
        """
        embedder_params = {"provider": "database", "model": model_name}
        embedder = OracleEmbeddings(conn=self.connection, params=embedder_params)
        embed = embedder.embed_query(query)
        return embed

    def load_records_as_documents(self, query: str, page_content_column: str) -> list[Document]:
        """
        Executes a SQL query on the database and returns results as langchain documents
        :param query:
        :param page_content_column:
        :return:
        """
        df = pd.read_sql(
            query,
            self.engine)
        docs = (
            DataFrameLoader(
                df,
                page_content_column=page_content_column
            )
            .load()
        )
        return docs




