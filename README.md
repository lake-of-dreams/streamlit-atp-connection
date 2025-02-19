# Streamlit Oracle Autonomous Database Connection

Streamlit ATP Connection is a Python library and Streamlit component that connects your Streamlit application to an Oracle Autonomous database. This integration enabled you to utilize Oracle database AI features in your application. You can run sql queries and load CSV data into the database using the utility functions provided by the library. The library can also be used for [load ONNX models](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/import-onnx-models-oracle-database-end-end-example.html) into the database and generate embeddings. 

[Streamlit](https://streamlit.io) is a framework to create web applications for machine learning and data science projects quickly and easily.

[Oracle Autonomous Database](https://www.oracle.com/in/autonomous-database/) a set of self-driving data management services providing several built-in AI capabilities. 

The Streamlit ATP Connection is built upon tools provided by [OCI Python SDK](https://github.com/oracle/oci-python-sdk) , the [python-oracledb](https://oracle.github.io/python-oracledb/) driver and [langchain](https://python.langchain.com/docs/concepts/) interfaces.

## Getting started

### Installation

You can install Streamlit ATP Connections with pip:

```shell
pip install -U git+https://github.com/lake-of-dreams/streamlit-atp-connection
```

### OCI credentials

In order to connect to the ATP instances using this library, You will need to configure the OCI credentials for your Streamlit app. 
You can also add the OCI credentials to the connection-specific section of your `secrets.toml`. For example, if you name your connection `atpconnection`, you can add the following in the secrets file:

    ```conf
    # .streamlit/secrets.toml

    [connections.atpconnection]
    oci_region="ap-mumbai-1"
    oci_user="ocid..xxx.xxx"
    oci_fingerprint="xx:xx:xx"
    oci_keyfile="/xx/x.pem"
    oci_tenancy="xx"
    ```

While running locally these credentials can be provided in a file (default location for OCI config file is ~/.oc/config).

### Using Streamlit ATP Connection

Following example provides usage of Streamlit ATP Connection in your Streamlit applications:

```python
import pathlib
from zipfile import ZipFile

import streamlit as st
import wget

from atp_connection import ATPConnection

conn = st.connection("atpconnection", type=ATPConnection)
atp_instances = conn.fetch_atp_instances("<compartment_id>")
for atp_id in atp_instances:
    st.write(f"ATP OCID: {atp_id}")
    st.markdown(atp_instances[atp_id].db_version)
    if atp_instances[atp_id].db_version == "23ai" :
        for profile in atp_instances[atp_id].connection_strings.profiles:
            conn.connect(atp_id,profile.value,"<db_user>","<db_password>")
            # Execute SQL
            st.table(conn.execute("select 1 from dual"))
            if profile.display_name.endswith("_tp"):
                with st.spinner("Loading csv"):
                    # Load CSV data
                    conn.load_csv("labels", True, 800, "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv",sep="\t",
                                header=0, index_col=0)
                    # Sample table data
                    conn.sample_data("labels")
                with st.spinner("Loading ONNX model"):
                    pathlib.Path("/tmp/model").mkdir(exist_ok=True)
                    wget.download("https://adwc4pm.objectstorage.us-ashburn-1.oci.customer-oci.com/p/VBRD9P8ZFWkKvnfhrWxkpPe8K03-JIoM5h_8EJyJcpE80c108fuUjg7R5L5O7mMZ/n/adwc4pm/b/OML-Resources/o/all_MiniLM_L12_v2_augmented.zip", "/tmp/model")
                    ZipFile("/tmp/model/all_MiniLM_L12_v2_augmented.zip").extractall("/tmp/model")
                    # Load ONNX model
                    conn.load_onnx_model("/tmp/model/all_MiniLM_L12_v2.onnx","mini_llm")
                with st.spinner("Generate embedding"):
                    # Generate embedding
                    st.markdown(conn.create_embedding("mini_llm","Hello World!"))
```

### Run example

```shell
python3 -m venv venv
. venv/bin/activate
pip install streamlit
pip install -e .
streamlit run examples/example.py
```