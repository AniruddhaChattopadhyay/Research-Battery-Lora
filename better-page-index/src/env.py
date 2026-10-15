"""Load the project .env file (API keys) into the environment.

The .env file in the project folder is the one place for the keys. Its values win over
variables that the shell already has, so a change in .env always takes effect.
"""
import os

from dotenv import load_dotenv

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(ENV_FILE, override=True)
