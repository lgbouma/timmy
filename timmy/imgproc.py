import numpy as np

def integer_shift_img(img, dx, dy):
    # zero-filled array at edges, otherwise shifted in x and y (where we follow
    # the standard numpy "x" and "y" dimension conventions) by the given dx/dy.

    assert len(img.shape)==2
    assert isinstance(dx, int)
    assert isinstance(dy, int)

    non = lambda s: s if s<0 else None
    mom = lambda s: max(0,s)

    shift_img = np.zeros_like(img)
    shift_img[mom(dy):non(dy), mom(dx):non(dx)] = (
        img[mom(-dy):non(-dy), mom(-dx):non(-dx)]
    )

    return shift_img
