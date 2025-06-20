def compute_centered_roi(shape, percent=0.3):
    """
    Compute a centered square/rectangular ROI based on a percentage of the image size.

    Parameters
    ----------
    shape : tuple
        (height, width) of the image.
    percent : float
        Fraction of the image size to include (e.g., 0.3 = 30%).

    Returns
    -------
    roi : tuple
        (row_start, row_end, col_start, col_end)
    """
    lines, columns = shape
    row_center = lines // 2
    col_center = columns // 2

    half_h = int((lines * percent) / 2)
    half_w = int((columns * percent) / 2)

    return (
        row_center - half_h,
        row_center + half_h,
        col_center - half_w,
        col_center + half_w,
    )