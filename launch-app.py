from streamlit.web import cli as stcli
import sys
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    sys.argv = ["streamlit", "run", "2_app/app.py", "--server.port", "8080", "--server.address", "127.0.0.1"]
    sys.exit(stcli.main())
