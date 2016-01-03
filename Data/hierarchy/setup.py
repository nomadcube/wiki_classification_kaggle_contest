from distutils.core import Extension, setup


hierarchy_module = Extension("_hierarchy", sources=['hierarchy_wrap.cxx', 'hierarchy.cpp'],)

setup(name="hierarchy",
      version=1.0,
      author="wumengling",
      ext_modules=[hierarchy_module],
      py_modules=["hierarchy"],)
