def RMSErrorNV(ys, y_hats):
    """
    Root mean squared. Not Vectorized (NV).
    """
    ysize = len(ys)
    assert ysize == len(y_hats), "ys and y_hats must be in same size."
    sum = 0
    for y, y_hat in zip(ys, y_hats):
        sum += (y - y_hat) ** 2
    if ysize == 1:
        return 0.5 * sum
    else:
        return 1/ysize * sum