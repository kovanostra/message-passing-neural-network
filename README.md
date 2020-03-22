### Requirements
Python 3.7.6

Run (only for tests at the moment)
```
numpy==1.17.4
pytorch=1.4.0
```

Build
```
tox==3.14.3
```

To run all tests and build the project, just cd to ~/protein-folding/ and run (with sudo if necessary)
```
tox
```

This will automatically create an artifact and place it in ~/protein-folding/.tox/dist/graph-to-graph-version.zip. The version can be specified in the setup.py. The contents of this folder are cleaned at the start of every new build.

### Entrypoint

Currently, there is no entrypoint. However, the code can be explored through the tests.

### Azure pipelines project

https://dev.azure.com/kovamos/protein-folding
