"""
Zotero generates long journal names. Abbreviate them in bulk to conform to the
AAS style guide.
"""

from collections import OrderedDict

mdict = OrderedDict({})

mdict['The Astrophysical Journal Letters'] = r'\apjl'
mdict['The Astrophysical Journal Supplement Series'] = r'\apjs'
mdict['The Astronomical Journal'] = r'\aj'
mdict['Annual Review of Astronomy and Astrophysics'] = r'\araa'
mdict['The Astrophysical Journal'] = r'\apj'
mdict['Publications of the Astronomical Society of the Pacific'] = r'\pasp'
mdict['Astronomy and Astrophysics'] = r'\aap'
mdict['Astronomy & Astrophysics'] = r'\aap'
mdict['Monthly Notices of the Royal Astronomical Society: Letters'] = r'\mnras:l'
mdict['Monthly Notices of the Royal Astronomical Society'] = r'\mnras'
mdict['Nature'] = r'\nat'

with open('../paper/bibliography.bib', 'r') as f:
    lines = f.readlines()

for k,v in mdict.items():
    for ix, l in enumerate(lines):
        if k in l:
            _newline = l.replace(k, v)
            lines[ix] = _newline

outpath = 'abbrevbib.bib'
with open(outpath, 'w') as f:
    f.writelines(lines)

print('made {}'.format(outpath))
