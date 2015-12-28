from distutils.core import setup, Extension

sparse_vector_module = Extension('_sparse_vector',
                                 sources=['sparse_vector_wrap.cxx', 'sparse_vector.cpp'], )


setup(name='sparse_vector',
      version='0.1',
      author='wumengling',
      description='sparse vector for gradient descent',
      ext_modules=[sparse_vector_module],
      py_modules=["sparse_vector"],
      )
