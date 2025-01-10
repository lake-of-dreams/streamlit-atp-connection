import io
import string
import random
from typing import Dict, Any, Optional
from zipfile import ZipFile
import pathlib

import oci

def write_files(file: io.BytesIO) -> str:
    random_path = random_string()
    dir_name = "/tmp/"+random_path
    pathlib.Path(dir_name).mkdir(exist_ok=True)
    ZipFile(file).extractall(dir_name)
    return dir_name

def random_string(min_length=10, max_length=15):
    length = random.randint(min_length, max_length)
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(length)
    )

def random_passwd(min_length=8, max_length=15):
    length = random.randint(min_length, max_length)
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase + "#$@")
        for _ in range(length)
    )

def extract_zip(input_zip):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def get_config_param(key: str, alt: str, config: Dict[str, Any]) -> Optional[Any]:
    return config.get(key) or config.get(key.upper()) or config.get(alt) or config.get(alt.upper())

def oci_config(config: Optional[Dict[str, Any]]) -> Optional[dict]:
    ociconfig = dict(oci.config.DEFAULT_CONFIG)
    if config is None:
        ociconfig = oci.config.from_file(oci.config.DEFAULT_LOCATION, oci.config.DEFAULT_PROFILE)
    else:
        region = get_config_param("oci_cli_region", "oci_region", config)
        user = get_config_param("oci_cli_user", "oci_user", config)
        fingerprint = get_config_param("oci_cli_fingerprint", "oci_fingerprint", config)
        keyfile = get_config_param("oci_cli_keyfile", "oci_keyfile", config)
        tenancy = get_config_param("oci_cli_tenancy", "oci_tenancy", config)
        if region is not None and user is not None and fingerprint is not None and keyfile is not None and tenancy is not None:
            ociconfig["region"] = region
            ociconfig["user"] = user
            ociconfig["fingerprint"] = fingerprint
            ociconfig["keyfile"] = keyfile
            ociconfig["tenancy"] = tenancy
            passphrase = get_config_param("oci_cli_passphrase", "oci_passphrase", config)
            if passphrase is not None:
                ociconfig["pass_phrase"] = passphrase

    oci.config.validate_config(ociconfig)
    return ociconfig

