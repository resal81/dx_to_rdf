

A script to analyze density plot generated by VMD plugin VolMap.

Sample run:
- `python dx_to_rdf.py --cmd analyze --dx data.dx --cache --plot --maxr 60 --limr 50 --minz -30 --maxz 30`
- By default: `maxr` is 60, `limr` is 40, `minz` is -30 and `maxz` is 30. So if these values are OK, you don't
  need to specify them.

Help:
- `python dx_to_rdf.py -h`
