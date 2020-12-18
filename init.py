import os

pwd = os.getcwd()
os.environ['PYTHONPATH'] = f'{pwd}:{pwd}/src' if not 'PYTHONPATH' in os.environ else \
    f'{os.environ["PYTHONPATH"]}:{pwd}:{pwd}/src'
