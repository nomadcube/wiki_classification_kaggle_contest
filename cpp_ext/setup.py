from distutils.core import setup, Extension

k_argmax_module = Extension('_k_argmax',
                            sources=['k_argmax_wrap.cxx', 'k_argmax.cpp'], )

setup(name='k_argmax',
      version=0.1,
      author='wumengling',
      ext_modules=[k_argmax_module],
      py_modules=["k_argmax"],
      )
