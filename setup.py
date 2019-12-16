from setuptools import setup

setup(name = "pynolca",

      version = "0.1.0",

      description = '''A python package for noise-resilient
                       online large-scale classification algorithm.''',

      author = "Xingke Chen",

      author_email = "chenxk1229@hotmail.com", 

      download_url = "http://small.sci.upc.edu.cn/paper.aspx",

      packages = ["pynolca"],

      install_requires = ["numpy >= 1.8.2",
                          "matplotlib >= 2.0.0",]

    )
