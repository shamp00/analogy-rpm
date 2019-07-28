import logging
import re
import json

def log_init(filename: str):
    logging.basicConfig(filename=filename, level=logging.INFO, format='')
#   logging.getLogger().addHandler(logging.StreamHandler())

def log(message: str = ''):
    print(message)
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')
    stripped = ansi_escape.sub('', message)
    logging.info(stripped)

def log_dict(d: dict):
    log(json.dumps(d, indent=2))
