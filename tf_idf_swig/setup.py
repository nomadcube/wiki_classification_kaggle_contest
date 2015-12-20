from distutils.core import setup, Extension

tf_idf_module = Extension('_tf_idf',
                           sources=['tf_idf_wrap.cxx', 'tf_idf.cpp'],)


setup(name='tf_idf',
      version='0.1',
      author='wumengling',
      description='used for wiki text mining',
      ext_modules=[tf_idf_module],
      py_modules=["tf_idf"],
      )
