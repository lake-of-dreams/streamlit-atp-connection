from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="streamlit_atp_connection",
    version="0.0.1",
    author="Ravi Singhal",
    author_email="ravi.r.singhal@oracle.com",
    description="Streamlit component that allows you to connect to Oracle Autonomous databases using OCI credentials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.10",
    install_requires=[
        # By definition, a Custom Component depends on Streamlit.
        # If your component has other Python dependencies, list
        # them here.
        "streamlit >= 0.63",
        "oci >= 2.141.1",
        "oracledb >= 2.5.1",
        "langchain_community >= 0.3.14",
        "langchain_core >= 0.3.29",
        "pandas >= 2.2.3"
    ],
    extras_require={
        "devel": [
            "wheel",
            "pytest==7.4.0",
            "playwright==1.48.0",
            "requests==2.31.0",
            "pytest-playwright-snapshot==1.0",
            "pytest-rerunfailures==12.0",
        ]
    }
)
