'''
make table of RVs
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from numpy import array as nparr

from glob import glob
import os, pickle

from timmy.paths import DATADIR, RESULTSDIR

def main():

    objname = 'TOI837'
    rvpath = os.path.join(DATADIR, 'spectra', 'RVs_20200624_clean.csv')

    outdir = os.path.join(RESULTSDIR, 'paper_tables')

    df = pd.read_csv(rvpath)

    outdf = pd.DataFrame(
        {'time': np.round(nparr(df.time-2450000),6),
         'mnvel': np.round(nparr(df.mnvel),1),
         'errvel': np.round(nparr(df.errvel),1),
         'tel': nparr(df.tel)
        }
    )

    outpath = os.path.join(outdir, f'{objname}_rv_data.tex')
    with open(outpath,'w') as tf:
        tf.write(outdf.to_latex(index=False, escape=False))

    print('wrote {:s}'.format(outpath))

    outcsv = os.path.join(outdir, f'{objname}_rv_data.csv')
    outdf.to_csv(outcsv, index=False, float_format='%.6f')
    print(f'also wrote {outcsv}')


    #
    # trim this tex output to remove the annoying first rows, so you
    # can \input{TOI837_rv_data.tex} from the table file.
    #
    with open(outpath,'r') as f:
        lines = f.readlines()
    midrule_ix = [ix for ix,l in enumerate(lines)
                  if r'\midrule' in l][0]
    bottomrule_ix = [ix for ix,l in enumerate(lines)
                     if r'\bottomrule' in l][0]
    sel_lines = lines[midrule_ix+1 : bottomrule_ix]
    # sel_lines = [l.replace(' \\\\', '') for l in sel_lines]

    with open(outpath, 'w') as f:
        f.writelines(sel_lines)
    print('made {}'.format(outpath))



if __name__=="__main__":
    main()
