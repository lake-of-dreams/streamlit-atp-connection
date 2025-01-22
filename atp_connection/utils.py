"""Helper methods for ATPConnection"""
import io
import re
import string
import random
from typing import Dict, Any, Optional
from zipfile import ZipFile
import pathlib

import oci

def write_files(file: io.BytesIO | bytes, zip_file : bool) -> str:
    random_path = random_string()
    dir_name = "/tmp/"+random_path
    pathlib.Path(dir_name).mkdir(exist_ok=True)
    if zip_file:
        ZipFile(file).extractall(dir_name)
        file_location = dir_name
    else:
        file_path = f"{dir_name}/{random_string()}"
        pathlib.Path(file_path).write_bytes(file)
        file_location = file_path
    return file_location

def random_string(min_length=10, max_length=15):
    length = random.randint(min_length, max_length)
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(length)
    )

def random_passwd(min_length=8, max_length=15):
    length = random.randint(min_length, max_length)
    match = False
    passwd = ""
    reg = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{"+f"{min_length},{max_length}"+"}$"
    pat = re.compile(reg)
    while not match:
        passwd = ''.join(
            random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase + "#$@")
            for _ in range(length)
        )
        match = re.search(pat, passwd)

    return passwd

def get_config_param(key: str, alt: str, config: Dict[str, Any]) -> Optional[Any]:
    return config.get(key) or config.get(key.upper()) or config.get(alt) or config.get(alt.upper())

def oci_config(**kwargs) -> Optional[dict]:
    oci_profile = kwargs.get("oci_profile")
    if oci_profile is None:
        oci_profile = "DEFAULT"

    ociconfig = oci_config_from_dict(kwargs)
    if ociconfig is None:
        config_from_kwargs = kwargs.get("config")
        if config_from_kwargs is not None:
            ociconfig = oci_config_from_dict(config_from_kwargs)
            if ociconfig is None:
                raise "invalid config provided"
        else:
            config_file_from_kwargs = kwargs.get("config_file")
            if config_file_from_kwargs is not None:
                ociconfig = oci.config.from_file(config_file_from_kwargs, oci_profile)
            else:
                ociconfig = oci.config.from_file(oci.config.DEFAULT_LOCATION, oci_profile)

    oci.config.validate_config(ociconfig)
    return ociconfig

def oci_config_from_dict(config: Optional[Dict[str, Any]]) -> dict[str, Any] | None:
    region = get_config_param("oci_cli_region", "oci_region", config)
    user = get_config_param("oci_cli_user", "oci_user", config)
    fingerprint = get_config_param("oci_cli_fingerprint", "oci_fingerprint", config)
    keyfile = get_config_param("oci_cli_keyfile", "oci_keyfile", config)
    tenancy = get_config_param("oci_cli_tenancy", "oci_tenancy", config)
    if region is not None and user is not None and fingerprint is not None and keyfile is not None and tenancy is not None:
        ociconfig = dict(oci.config.DEFAULT_CONFIG)
        ociconfig["region"] = region
        ociconfig["user"] = user
        ociconfig["fingerprint"] = fingerprint
        ociconfig["keyfile"] = keyfile
        ociconfig["tenancy"] = tenancy
        passphrase = get_config_param("oci_cli_passphrase", "oci_passphrase", config)
        if passphrase is not None:
            ociconfig["pass_phrase"] = passphrase
        return ociconfig
    else:
        return None


