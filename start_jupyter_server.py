import os

os.system("jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.disable_check_xsrf=True \
  --port=8888 \
  --NotebookApp.port_retries=0")

# To start for in Thesis Directory:
# cd /data/GoogleDrive/UNI/AUTH/COURSES/10th\ Semester\ -\ Thesis/Courses/ && jupyter notebook
