import os

os.system("jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.disable_check_xsrf=True \
  --port=8888 \
  --NotebookApp.port_retries=0")
