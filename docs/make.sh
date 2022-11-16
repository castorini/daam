rm daam.*rst daam.rst
rm -rf _build
sphinx-apidoc -f -o . ../daam
make html
make github

