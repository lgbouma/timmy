def chisq(y_mod, y_obs, y_err):
    return np.sum( (y_mod - y_obs )**2 / y_err**2 )


def bic(chisq, k, n):
    """
    BIC = χ^2 + k log n, for k the number of free parameters, and n the
    number of data points.
    """
    return chisq + k*np.log(n)


def get_bic(m, ydict, outdir):

    y_obs = ydict['y_obs']
    y_err = m.y_err
    y_mod = ydict['y_mod_tra'] + ydict['y_mod_orb'] + ydict['y_mod_rot']

    χ2 = chisq(y_mod, y_obs, y_err)

    k = None
    _N = int(m.modelid.split('_')[1][0])
    _M = int(m.modelid.split('_')[2][0])

    k_tra = 7
    k_Porb = 2*_N # amplitudes
    k_Prot = 2*_M + 2 # amplitudes, plus period and phase
    k = k_tra + k_Porb + k_Prot

    n = len(y_obs)
    BIC = bic( χ2, k, n )

    dof = n-k

    msg = (
        '{}: Ndata = {:d}, Nparam = {:d}, χ2 = {:.1f}, redχ2 = {:.2f}, BIC = {:.1f}'.
        format(m.modelid, n, k, χ2, χ2/dof, BIC)
    )
    print(42*'=')
    print(msg)
    print(42*'=')

    bicdict = {
        'modelid': m.modelid,
        'N': _N,
        'M': _M,
        'Ndata': n,
        'Nparam': k,
        'chisq': χ2,
        'redchisq': χ2/dof,
        'BIC': BIC
    }
    pklpath = os.path.join(outdir, m.modelid+'_bicdict.pkl')
    with open(pklpath, 'wb') as buff:
        pickle.dump(bicdict, buff)
    print('Wrote {}'.format(pklpath))



