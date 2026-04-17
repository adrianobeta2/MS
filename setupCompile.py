from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        ["servidor.py", "treinar_modelo_new.py", "monitorar_movimentos.py", "machine_guide.py"],
        compiler_directives={'language_level': "3"}
    )
)

