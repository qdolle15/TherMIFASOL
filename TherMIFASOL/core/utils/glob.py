import os

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

def create_unique_folder(path: str) -> str:
    """
    Create a unique folder from a given path.
    If the folder already exists, append a numeric suffix (_1, _2, ...) 
    until a free name is found. Then create the folder.

    Parameters
    ----------
    path : str
        Base path of the folder to create.

    Returns
    -------
    str
        Path of the folder that was actually created.
    """
    folder_path = os.path.abspath(path)
    counter = 1

    # Increment until a free path is found
    while os.path.exists(folder_path):
        folder_path = f"{os.path.abspath(path)}_{counter}"
        counter += 1

    os.makedirs(folder_path)
    return folder_path