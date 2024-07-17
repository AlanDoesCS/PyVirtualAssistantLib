#!/bin/bash
./build.sh
python3 -m twine upload --repository testpypi dist/*