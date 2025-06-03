import numpy as np
from scipy.fft import fft2, ifft2
from scipy.integrate import simpson as simps
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class RegularizedThermalGradient:
    """
    This class is designed to compute a gradient on 2D data using a Fourier series decomposition. 
    It then regularizes an ill-posed problem to compute a function that best approximates the data while penalizing oscillations. 
    This penalty is characterized by the alpha coefficient.
    """
    
    def __init__(
            self,
            raw_image:np.array,
            crop_data:tuple,
            harmonics_X:int,
            harmonics_Y:int,
            scale:float,
            manual_alpha:float,
            optimized:bool,
            ):
        """
        Initialize the RegularizedThermalGradient class with raw image data and parameters.

        Parameters:
            raw_image (np.ndarray): The raw image on which the gradient is to be computed.
            crop_data (tuple): (X0, Y0, L, H) are the anchor points and dimensions of the cropping window.
            harmonics_X (int): The number of harmonics in the Fourier series decomposition along the X axis.
            harmonics_Y (int): The number of harmonics in the Fourier series decomposition along the Y axis.
            scale (float): The scaling factor [mm/pixel].
            manual_alpha (float): The regularization penalty factor in the manual gradient formulation.
        """
        
        # Inputs
        self.raw_image=raw_image
        self.crop_data=crop_data
        self.harmonics_X=harmonics_X
        self.harmonics_Y=harmonics_Y
        self.scale=scale
        self.manual_alpha=manual_alpha

        # Tools
        self.print_state = False
        self.filled_image, self.mask_filled=self.fill_raw_image()

        # Domain
        self.Ly = np.asarray(self.filled_image.shape)[0]*scale 
        self.Lx = np.asarray(self.filled_image.shape)[1]*scale
        self.x = np.linspace(-self.Lx/2, self.Lx/2, self.filled_image.shape[1])
        self.y = np.linspace(-self.Ly/2, self.Ly/2, self.filled_image.shape[0])
        self.grid_X, self.grid_Y = np.meshgrid(self.x, self.y)

        # Results
        self.Fourier_coefficients = None
        self.grad_Fourier_from = None
        self.grad_regularized = None

        if optimized:  # AI enhancement reduces calculation time from 50 to 10 seconds.
            self.get_Fourier_coefficients_vectorized()
            self.get_grad_Fourier_formulation_vectorized()
            self.get_regularized_grad_vectorized(manual_alpha)
        
        else:
            self.get_Fourier_coefficients()
            self.get_grad_Fourier_formulation()
            self.get_regularized_grad(manual_alpha)       

        self.thermal_gradient_norm, self.thermal_gradient_orientation =\
              self.get_thermal_gradient_norm_and_orientation()
        
        self.thermal_gradient_norm = self.resize_to_original(self.thermal_gradient_norm)
        self.thermal_gradient_norm *= 1000  # K/m
        self.thermal_gradient_orientation = self.resize_to_original(self.thermal_gradient_orientation)

    # Tools
    def fill_raw_image(self, fac_exp: float = 0.05, threshold: float = 500) -> tuple:
        """
        Adapt the raw image to have periodicity-like shape for Fourier decomposition.

        Parameters:
        fac_exp (float): Extension factor of the domain.
        threshold (float): Low limit to reach at the boundary to make the image periodic.

        Returns:
        tuple: Extended periodicity-like image and mask of filled data.
        """

        if self.crop_data == (0, 0, 0, 0):
            # Automatic crop to the smallest domain containing all the data 
            xpos, ypos = np.where(~np.isnan(self.raw_image))
            xmin, xmax = np.min(xpos), np.max(xpos)
            ymin, ymax = np.min(ypos), np.max(ypos)
            self.crop_data = (xmin, ymin, xmax-xmin, ymax-ymin)
        else:
            xmin, ymin, x_dist, y_dist = self.crop_data
            xmax, ymax = xmin + x_dist, ymin + y_dist

        # Domain extention with NaN values related to the 'fac_exp' extension factor
        res = self.raw_image[xmin:xmax, ymin:ymax]
        res = np.hstack((res, np.full((res.shape[0], int(res.shape[1]*fac_exp)), fill_value=np.nan)))
        res = np.hstack((np.full((res.shape[0], int(res.shape[1]*fac_exp)), fill_value=np.nan), res))

        res = np.vstack((res, np.full((int(res.shape[0]*fac_exp), res.shape[1]), fill_value=np.nan)))
        res = np.vstack((np.full((int(res.shape[0]*fac_exp), res.shape[1]), fill_value=np.nan), res))

        # Mask of the unfilled values (true values)
        mask_filled_data = np.isnan(res)

        # Linear filling of the NaN values until a low threshold
        ## Managing columns first
        for col in range(res.shape[1]):        
            start=0  # first line
            end=res.shape[0]  # last line

            pos_=np.where(~np.isnan(res[:,col]))[0]  # data detection
            last_line=0
            if len(pos_) < 10:
                res[:,col]=np.nan
            else:  
                for cpt, lin in enumerate(pos_):
                    if last_line == 0 and lin !=0 :  # between upper boundary and data
                        res[start:lin,col]=np.linspace(
                            threshold, res[lin,col], lin-last_line
                            )

                        last_line = lin

                    if cpt == len(pos_)-1 and lin!=end:  # between lower boundary and data
                        res[lin+1:end,col]=np.linspace(
                            res[lin,col], threshold, end-lin-1
                            )

                    if lin - last_line > 1:  # between data
                        res[last_line:lin,col] = np.linspace(
                            res[last_line,col], res[lin,col], lin-last_line
                            )
                        last_line=lin
    
                    else:
                        last_line=lin

        ## Managing lines then
        for lin in range(res.shape[0]):

            end_ = res.shape[1]
            min_ = np.min(np.where(~np.isnan(res[lin,:]) == True))
            max_ = np.max(np.where(~np.isnan(res[lin,:]) == True))
            # Fill NaN values before the first non-NaN value
            if min_ > 0:
                res[lin, :min_] = np.linspace(threshold, res[lin, min_], min_)
            
            # Fill NaN values after the last non-NaN value
            if max_ < end_ - 1:
                res[lin, max_ + 1:] = np.linspace(res[lin, max_], threshold, end_ - max_ - 1)
            
            # Fill NaN values between non-NaN values
            nan_indices = np.where(np.isnan(res[lin, :]))[0]
            for i in range(len(nan_indices)):
                if nan_indices[i] > min_ and nan_indices[i] < max_:
                    prev_non_nan = nan_indices[i] - 1
                    next_non_nan = nan_indices[i]
                    while next_non_nan < end_ and np.isnan(res[lin, next_non_nan]):
                        next_non_nan += 1
                    if next_non_nan < end_:
                        res[lin, prev_non_nan + 1:next_non_nan] = np.linspace(
                            res[lin, prev_non_nan], res[lin, next_non_nan], next_non_nan - prev_non_nan - 1
                        )
                        
        return res, mask_filled_data

    def resize_to_original(self, image_to_resize: np.ndarray) -> np.ndarray:
        """
        Resize the extended image back to its original dimensions, excluding the extended lines.

        Parameters:
        image_to_resize (np.ndarray): The extended periodicity-like image.

        Returns:
        np.ndarray: Resized image and mask.
        """
        # Extract the original crop data
        xmin, ymin, x_dist, y_dist = self.crop_data

        # Calculate the original dimensions
        original_height = self.raw_image.shape[0]
        original_width = self.raw_image.shape[1]

        # Create an empty array with the original dimensions
        resized_image = np.full((original_height, original_width), np.nan)
        resized_mask = np.full((original_height, original_width), True)

        # Calculate the dimensions of the extended image
        extended_height = image_to_resize.shape[0]
        extended_width = image_to_resize.shape[1]

        # Calculate the start and end indices for the original data region
        start_x = int((extended_height - x_dist) / 2)
        end_x = start_x + x_dist
        start_y = int((extended_width - y_dist) / 2)
        end_y = start_y + y_dist

        # Place the original data region back into the original dimensions
        resized_image[xmin:xmin + x_dist, ymin:ymin + y_dist] = image_to_resize[start_x:end_x, start_y:end_y]

        return resized_image

    def solve_regularization_strong_equation(self, Aij: float, Bij: float, Cij: float, Dij: float, i: int, j: int, alpha: float) -> np.ndarray:
        """
        Solve the strong formulation of the regularization's problem.

        Parameters:
        Aij (float): Fourier coefficient Aij.
        Bij (float): Fourier coefficient Bij.
        Cij (float): Fourier coefficient Cij.
        Dij (float): Fourier coefficient Dij.
        i (int): Harmonic index along the X axis.
        j (int): Harmonic index along the Y axis.
        alpha (float): Regularization penalty factor.

        Returns:
        np.ndarray: Solution to the regularization problem.
        """
        
        I = i*np.pi/(self.Lx/2)
        J = j*np.pi/(self.Ly/2)

        M = np.array([
            [1 + alpha*(I)**2, 0, 0, 0, 0, 0, 0, -alpha*I*J],
            [0, 1 + alpha*(I)**2, 0, 0, 0, 0, +alpha*I*J, 0],
            [0, 0, 1 + alpha*(I)**2, 0, 0, +alpha*I*J, 0, 0],
            [0, 0, 0, 1 + alpha*(I)**2, -alpha*I*J, 0, 0, 0],
            [0, 0, 0, -alpha*I*J, 1 + alpha*(J)**2, 0, 0, 0],
            [0, 0, +alpha*I*J, 0, 0, 1 + alpha*(J)**2, 0, 0],
            [0, +alpha*I*J, 0, 0, 0, 0, 1 + alpha*(J)**2, 0],
            [-alpha*I*J, 0, 0, 0, 0, 0, 0, 1 + alpha*(J)**2],
        ])  

        right_side = np.array([
            I*Cij,
            I*Dij,
            -I*Aij,
            -I*Bij,
            J*Bij,
            -J*Aij,
            J*Dij,
            -J*Cij,
        ])

        res = np.linalg.solve(M, right_side)

        return res

    def solve_regularization_strong_equation_vectorized(self, alpha: float) -> np.ndarray:
        """
        Vectorized version of the strong regularization solver.

        Parameters:
        alpha (float): Regularization penalty factor.

        Returns:
        np.ndarray: Solution to the regularized system of shape (8, Hx, Hy)
        """
        pi = np.pi
        Hx, Hy = self.harmonics_X, self.harmonics_Y

        # Harmonic indices
        i = np.arange(Hx)[:, None] * pi / (self.Lx / 2)  # (Hx, 1)
        j = np.arange(Hy)[None, :] * pi / (self.Ly / 2)  # (1, Hy)
        I2 = i**2
        J2 = j**2
        IJ = i * j

        # Fourier coefficients
        A, B, C, D = self.Fourier_coefficients  # shape (4, Hx, Hy)

        # Build RHS (8, Hx, Hy)
        RHS = np.empty((8, Hx, Hy))
        RHS[0] = i * C
        RHS[1] = i * D
        RHS[2] = -i * A
        RHS[3] = -i * B
        RHS[4] = j * B
        RHS[5] = -j * A
        RHS[6] = j * D
        RHS[7] = -j * C

        # Build M (8, 8, Hx, Hy)
        M = np.zeros((8, 8, Hx, Hy))
        M[0, 0] = 1 + alpha * I2
        M[1, 1] = 1 + alpha * I2
        M[2, 2] = 1 + alpha * I2
        M[3, 3] = 1 + alpha * I2
        M[4, 4] = 1 + alpha * J2
        M[5, 5] = 1 + alpha * J2
        M[6, 6] = 1 + alpha * J2
        M[7, 7] = 1 + alpha * J2
        M[0, 7] = -alpha * IJ
        M[1, 6] = +alpha * IJ
        M[2, 5] = +alpha * IJ
        M[3, 4] = -alpha * IJ
        M[4, 3] = -alpha * IJ
        M[5, 2] = +alpha * IJ
        M[6, 1] = +alpha * IJ
        M[7, 0] = -alpha * IJ

        M_reshaped = M.reshape(-1, 8, 8)        # (Hx*Hy, 8, 8)
        RHS_reshaped = RHS.reshape(-1, 8)       # (Hx*Hy, 8)
        solution = np.linalg.solve(M_reshaped, RHS_reshaped)     # (Hx*Hy, 8)
        return solution.T.reshape(8, Hx, Hy)                      # (8, Hx, Hy)

    def L_curve_terms(self, alpha: float = None) -> tuple:
        """
        Compute the derivatives of the regularized formulation to find the optimal alpha.

        Parameters:
        alpha (float): Regularization penalty factor. If None, use manual_alpha.

        Returns:
        tuple: Derivatives of the regularized formulation.
        """

        f_x = np.zeros_like(self.filled_image, dtype=np.float64)
        f_y = np.zeros_like(self.filled_image, dtype=np.float64)
        dfx_dx = np.zeros_like(self.filled_image, dtype=np.float64)
        dfy_dy = np.zeros_like(self.filled_image, dtype=np.float64)
        dfx_dy = np.zeros_like(self.filled_image, dtype=np.float64)
        dfy_dx = np.zeros_like(self.filled_image, dtype=np.float64)

        for hi in range(self.harmonics_X):
            for hj in range(self.harmonics_Y):

                Aij, Bij, Cij, Dij = self.Fourier_coefficients[:,hi,hj]
                Axij, Bxij, Cxij, Dxij, Ayij, Byij, Cyij, Dyij = self.solve_regularization_strong_equation(Aij, Bij, Cij, Dij, hi, hj, alpha)
                        
                # F solution
                term1x = Axij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2x = Bxij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3x = Cxij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4x = Dxij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))

                term1y = Ayij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2y = Byij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3y = Cyij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4y = Dyij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                
                f_x+=term1x+term2x+term3x+term4x
                f_y+=term1y+term2y+term3y+term4y

                # dF solution
                sup_i = (hi*np.pi/(self.Lx/2))
                sup_j = (hj*np.pi/(self.Ly/2))

                term1x = -Axij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2x = -Bxij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3x = +Cxij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4x = +Dxij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))

                term1y = -Ayij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2y = +Byij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3y = -Cyij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4y = +Dyij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                
                dfx_dx+=sup_i*(term1x+term2x+term3x+term4x)
                dfy_dy+=sup_j*(term1y+term2y+term3y+term4y)

                # dF crossed solution
                term1x = +Axij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2x = -Bxij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3x = -Cxij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4x = +Dxij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))

                term1y = +Ayij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2y = -Byij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3y = -Cyij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4y = +Dyij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                
                dfx_dy+=sup_i*sup_j*(term1x+term2x+term3x+term4x)
                dfy_dx+=sup_i*sup_j*(term1y+term2y+term3y+term4y)
        
        return f_x, f_y, dfx_dx, dfy_dy, dfx_dy, dfy_dx

    def L_curve_terms_vectorized(self, alpha: float = None) -> tuple:
        """
        Compute the derivatives of the regularized formulation to find the optimal alpha.

        Parameters:
        alpha (float): Regularization penalty factor. If None, use manual_alpha.

        Returns:
        tuple: Derivatives of the regularized formulation.
        """
        if self.print_state:
            print("\n\nComputing L-curve terms (vectorized)...\n")

        if alpha is None:
            eff_alpha = self.manual_alpha
        else:
            eff_alpha = alpha

        f_x = np.zeros_like(self.filled_image, dtype=np.float64)
        f_y = np.zeros_like(self.filled_image, dtype=np.float64)
        dfx_dx = np.zeros_like(self.filled_image, dtype=np.float64)
        dfy_dy = np.zeros_like(self.filled_image, dtype=np.float64)
        dfx_dy = np.zeros_like(self.filled_image, dtype=np.float64)
        dfy_dx = np.zeros_like(self.filled_image, dtype=np.float64)

        # Precompute the sine and cosine terms
        hi = np.arange(self.harmonics_X)[:, np.newaxis, np.newaxis, np.newaxis]
        hj = np.arange(self.harmonics_Y)[np.newaxis, :, np.newaxis, np.newaxis]

        # Reshape grid_X and grid_Y to include additional dimensions for harmonics
        grid_X_reshaped = self.grid_X[np.newaxis, np.newaxis, :, :]
        grid_Y_reshaped = self.grid_Y[np.newaxis, np.newaxis, :, :]

        cos_X = np.cos(hi * grid_X_reshaped * np.pi / (self.Lx / 2))
        sin_X = np.sin(hi * grid_X_reshaped * np.pi / (self.Lx / 2))
        cos_Y = np.cos(hj * grid_Y_reshaped * np.pi / (self.Ly / 2))
        sin_Y = np.sin(hj * grid_Y_reshaped * np.pi / (self.Ly / 2))

        # Precompute the scaling factors
        sup_i = hi * np.pi / (self.Lx / 2)
        sup_j = hj * np.pi / (self.Ly / 2)

        # Extract Fourier coefficients
        Aij = self.Fourier_coefficients[0]
        Bij = self.Fourier_coefficients[1]
        Cij = self.Fourier_coefficients[2]
        Dij = self.Fourier_coefficients[3]

        # Solve the regularization equation for all harmonics
        Axij, Bxij, Cxij, Dxij, Ayij, Byij, Cyij, Dyij = np.array([
            self.solve_regularization_strong_equation(Aij[hi, hj], Bij[hi, hj], Cij[hi, hj], Dij[hi, hj], hi, hj, eff_alpha)
            for hi in range(self.harmonics_X)
            for hj in range(self.harmonics_Y)
        ]).reshape((8, self.harmonics_X, self.harmonics_Y)).transpose(1, 2, 0)

        # Compute the terms for f_x and f_y
        term1x = Axij[:, :, np.newaxis, np.newaxis] * cos_X * cos_Y
        term2x = Bxij[:, :, np.newaxis, np.newaxis] * cos_X * sin_Y
        term3x = Cxij[:, :, np.newaxis, np.newaxis] * sin_X * cos_Y
        term4x = Dxij[:, :, np.newaxis, np.newaxis] * sin_X * sin_Y

        term1y = Ayij[:, :, np.newaxis, np.newaxis] * cos_X * cos_Y
        term2y = Byij[:, :, np.newaxis, np.newaxis] * cos_X * sin_Y
        term3y = Cyij[:, :, np.newaxis, np.newaxis] * sin_X * cos_Y
        term4y = Dyij[:, :, np.newaxis, np.newaxis] * sin_X * sin_Y

        f_x = np.sum(term1x + term2x + term3x + term4x, axis=(0, 1))
        f_y = np.sum(term1y + term2y + term3y + term4y, axis=(0, 1))

        # Compute the terms for dfx_dx and dfy_dy
        term1x = -Axij[:, :, np.newaxis, np.newaxis] * sin_X * cos_Y
        term2x = -Bxij[:, :, np.newaxis, np.newaxis] * sin_X * sin_Y
        term3x = Cxij[:, :, np.newaxis, np.newaxis] * cos_X * cos_Y
        term4x = Dxij[:, :, np.newaxis, np.newaxis] * cos_X * sin_Y

        term1y = -Ayij[:, :, np.newaxis, np.newaxis] * cos_X * sin_Y
        term2y = Byij[:, :, np.newaxis, np.newaxis] * cos_X * cos_Y
        term3y = -Cyij[:, :, np.newaxis, np.newaxis] * sin_X * sin_Y
        term4y = Dyij[:, :, np.newaxis, np.newaxis] * sin_X * cos_Y

        dfx_dx = np.sum(sup_i[:, :, np.newaxis, np.newaxis] * (term1x + term2x + term3x + term4x), axis=(0, 1))
        dfy_dy = np.sum(sup_j[:, :, np.newaxis, np.newaxis] * (term1y + term2y + term3y + term4y), axis=(0, 1))

        # Compute the terms for dfx_dy and dfy_dx
        term1x = Axij[:, :, np.newaxis, np.newaxis] * sin_X * sin_Y
        term2x = -Bxij[:, :, np.newaxis, np.newaxis] * sin_X * cos_Y
        term3x = -Cxij[:, :, np.newaxis, np.newaxis] * cos_X * sin_Y
        term4x = Dxij[:, :, np.newaxis, np.newaxis] * cos_X * cos_Y

        term1y = Ayij[:, :, np.newaxis, np.newaxis] * sin_X * sin_Y
        term2y = -Byij[:, :, np.newaxis, np.newaxis] * sin_X * cos_Y
        term3y = -Cyij[:, :, np.newaxis, np.newaxis] * cos_X * sin_Y
        term4y = Dyij[:, :, np.newaxis, np.newaxis] * cos_X * cos_Y

        dfx_dy = np.sum(sup_i[:, :, np.newaxis, np.newaxis] * sup_j[:, :, np.newaxis, np.newaxis] * (term1x + term2x + term3x + term4x), axis=(0, 1))
        dfy_dx = np.sum(sup_i[:, :, np.newaxis, np.newaxis] * sup_j[:, :, np.newaxis, np.newaxis] * (term1y + term2y + term3y + term4y), axis=(0, 1))

        return f_x, f_y, dfx_dx, dfy_dy, dfx_dy, dfy_dx

    def L_curve(self, alpha_range: np.ndarray) -> tuple:
        """
        Compute the L-curve to find the optimal alpha value.

        Parameters:
        alpha_range (np.ndarray): Range of alpha values to evaluate.

        Returns:
        tuple: Optimal alpha value, L-curve values, and J-curve values.
        """
        if self.print_state:
            print("\n\nL-curve processing...\n")
        grad_fourier_X, grad_fourier_Y=self.grad_Fourier_from

        J_alpha=[]
        L_alpha=[]
        for alpha in tqdm(alpha_range, desc="Processing alpha"):
            regul_fx, regul_fy,\
            regul_dfxdx, regul_dfydy,\
            regul_dfxdy, regul_dfydx,\
                =self.L_curve_terms(alpha)
            
            J_alpha.append(
                0.5*simps(simps((regul_fx-grad_fourier_X)**2 + (regul_fy-grad_fourier_Y)**2, self.x), self.y)
            )
            L_alpha.append(
                0.5*simps(simps((regul_dfxdx)**2 + (regul_dfydy)**2 + 2*regul_dfxdy*regul_dfydx, self.x), self.y)
            )

        L_alpha=np.asarray(L_alpha)
        J_alpha=np.asarray(J_alpha)
        alpha_min=alpha_range[np.argmin(np.sqrt(L_alpha**2 + J_alpha**2))]

        return alpha_min, L_alpha, J_alpha


    # Computation
    def get_Fourier_coefficients(self) -> np.ndarray:
        """
        Compute the Fourier coefficients using numerical methods for integration.

        Returns:
        np.ndarray: Fourier coefficients.
        """
        if self.print_state:
            print("\n\nFourier coefficients computation...\n")
        
        # Initialization of the results matrix
        res=np.zeros((4, self.harmonics_X, self.harmonics_Y))
        for hi in tqdm(range(self.harmonics_X), desc="Processing harmonics X"):
            for hj in tqdm(range(self.harmonics_Y), desc="Processing harmonics_Y", leave=False):

                # Boundaries conditions
                eta_ij = 1 
                if hi == 0 and hj == 0:
                    eta_ij = 1 / 4
                elif hi != hj and hi * hj == 0:
                    eta_ij = 1 / 2

                # Integrand matrix
                cos_i = np.cos(hi * np.pi * self.grid_X / (self.Lx/2))
                cos_j = np.cos(hj * np.pi * self.grid_Y / (self.Ly/2))
                sin_i = np.sin(hi * np.pi * self.grid_X / (self.Lx/2))
                sin_j = np.sin(hj * np.pi * self.grid_Y / (self.Ly/2))
                
                # Fourier coefficients
                Aij = (4*eta_ij/(self.Lx*self.Ly))*simps(simps(self.filled_image*cos_j*cos_i,self.x),self.y)
                Bij = (4*eta_ij/(self.Lx*self.Ly))*simps(simps(self.filled_image*cos_i*sin_j,self.x),self.y)
                Cij = (4*eta_ij/(self.Lx*self.Ly))*simps(simps(self.filled_image*sin_i*cos_j,self.x),self.y)
                Dij = (4*eta_ij/(self.Lx*self.Ly))*simps(simps(self.filled_image*sin_i*sin_j,self.x),self.y)

                # Stacking
                res[:,hi,hj]=Aij, Bij, Cij, Dij
        self.Fourier_coefficients = res
        return res

    def get_Fourier_coefficients_vectorized(self) -> np.ndarray:
        """Chat GPT, version optimisée de 'get_Fourier_coefficients' """
        if self.print_state:
            print("\n\nFourier coefficients computation...\n")
        # Correction du broadcasting : on reshape les grilles pour qu'elles soient compatibles
        kx = np.arange(self.harmonics_X)[:, None, None]  # (H_X, 1, 1)
        ky = np.arange(self.harmonics_Y)[None, :, None]  # (1, H_Y, 1)

        # Reshape pour broadcasting
        grid_X_b = self.grid_X[None, :, :]  # (1, Ny, Nx)
        grid_Y_b = self.grid_Y[:, None, :]  # (Ny, 1, Nx)

        # Calcul des bases trigonométriques
        cos_kx = np.cos(kx * np.pi * grid_X_b / (self.Lx/2))  # (H_X, Ny, Nx)
        sin_kx = np.sin(kx * np.pi * grid_X_b / (self.Lx/2))  # (H_X, Ny, Nx)
        cos_ky = np.cos(ky * np.pi * grid_Y_b / (self.Ly/2))  # (Ny, H_Y, Nx)
        sin_ky = np.sin(ky * np.pi * grid_Y_b / (self.Ly/2))  # (Ny, H_Y, Nx)

        # On transpose pour obtenir les mêmes dimensions (H_Y, Ny, Nx)
        cos_ky = np.transpose(cos_ky, (1, 0, 2))
        sin_ky = np.transpose(sin_ky, (1, 0, 2))

        # Initialisation du tableau de résultats
        res = np.zeros((4, self.harmonics_X, self.harmonics_Y))

        # Remplissage par double boucle (toujours nécessaire pour croiser X/Y)
        for i in range(self.harmonics_X):
            for j in range(self.harmonics_Y):
                eta_ij = 1
                if i == 0 and j == 0:
                    eta_ij = 1 / 4
                elif i != j and i * j == 0:
                    eta_ij = 1 / 2

                norm = 4 * eta_ij / (self.Lx * self.Ly)

                res[0, i, j] = norm * simps(simps(self.filled_image * cos_kx[i] * cos_ky[j], self.x), self.y)
                res[1, i, j] = norm * simps(simps(self.filled_image * cos_kx[i] * sin_ky[j], self.x), self.y)
                res[2, i, j] = norm * simps(simps(self.filled_image * sin_kx[i] * cos_ky[j], self.x), self.y)
                res[3, i, j] = norm * simps(simps(self.filled_image * sin_kx[i] * sin_ky[j], self.x), self.y)

        self.Fourier_coefficients = res
        return res


    def get_Fourier_formulation(self) -> np.ndarray:
        """
        Construct the Fourier series representation of the image.

        Returns:
        np.ndarray: Fourier series representation of the image.
        """
        if self.print_state:
            print("\n\nFourier signal formulation computation...\n")

        res = np.zeros_like(self.filled_image, dtype=np.float64)
        for hi in tqdm(range(self.harmonics_X), desc="Processing harmonics X"):
            for hj in tqdm(range(self.harmonics_Y), desc="Processing harmonics_Y", leave=False):

                Aij, Bij, Cij, Dij = self.Fourier_coefficients[:,hi,hj]

                term1 = Aij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2 = Bij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3 = Cij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4 = Dij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                
                res+=term1+term2+term3+term4
        
        return res
    
    def get_Fourier_formulation_vectorized(self) -> np.ndarray:
        """
        Reconstruct the field from the Fourier coefficients using the analytic Fourier series.
        Version 100 % vectorisée (sans boucle explicite).
        """
        if self.print_state:
            print("\n\nFourier signal formulation computation (fully vectorized)...\n")

        # Coefficients (4, Hx, Hy)
        A, B, C, D = self.Fourier_coefficients

        Hx, Hy = self.harmonics_X, self.harmonics_Y
        Ny, Nx = self.filled_image.shape

        # Indices
        kx = np.arange(Hx)
        ky = np.arange(Hy)

        # Pré-calcul des bases trigonométriques
        cos_kx = np.cos(kx[:, None] * np.pi * self.grid_X.flatten()[None, :] / (self.Lx / 2))  # (Hx, Ny*Nx)
        sin_kx = np.sin(kx[:, None] * np.pi * self.grid_X.flatten()[None, :] / (self.Lx / 2))  # (Hx, Ny*Nx)
        cos_ky = np.cos(ky[:, None] * np.pi * self.grid_Y.flatten()[None, :] / (self.Ly / 2))  # (Hy, Ny*Nx)
        sin_ky = np.sin(ky[:, None] * np.pi * self.grid_Y.flatten()[None, :] / (self.Ly / 2))  # (Hy, Ny*Nx)

        # Reshape pour multiplication : (Hx, 1, Ny*Nx), (1, Hy, Ny*Nx)
        cos_kx = cos_kx[:, None, :]     # (Hx, 1, N)
        sin_kx = sin_kx[:, None, :]
        cos_ky = cos_ky[None, :, :]     # (1, Hy, N)
        sin_ky = sin_ky[None, :, :]

        # Produits trigonométriques (Hx, Hy, N)
        coscos = cos_kx * cos_ky
        cossin = cos_kx * sin_ky
        sincos = sin_kx * cos_ky
        sinsin = sin_kx * sin_ky

        # Combinaison linéaire avec les coefficients de Fourier (broadcast sur N)
        total = (
            (A[:, :, None] * coscos) +
            (B[:, :, None] * cossin) +
            (C[:, :, None] * sincos) +
            (D[:, :, None] * sinsin)
        )  # shape (Hx, Hy, N)

        # Somme sur tous les modes Hx, Hy → reconstruction (N,)
        res_flat = total.sum(axis=(0, 1))  # (N,)

        # Reconstruction de la forme originale
        return res_flat.reshape(Ny, Nx)


    def get_grad_Fourier_formulation(self) -> tuple:
        """
        Compute the derivative of the Fourier series signal.

        Returns:
        tuple: Derivatives of the Fourier series signal.
        """
        if self.print_state:
            print("\n\nFirst derivative of Fourier signal formulation...\n")

        DT_dx = np.zeros_like(self.filled_image, dtype=np.float64)
        DT_dy = np.zeros_like(self.filled_image, dtype=np.float64)

        for hi in tqdm(range(self.harmonics_X), desc="Processing harmonics X"):
            for hj in tqdm(range(self.harmonics_Y), desc="Processing harmonics_Y", leave=False):

                Aij, Bij, Cij, Dij = self.Fourier_coefficients[:,hi,hj]

                sup_i = (hi*np.pi/(self.Lx/2))
                sup_j = (hj*np.pi/(self.Ly/2))

                term1x = -Aij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2x = -Bij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3x = Cij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4x = Dij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))

                term1y = -Aij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2y = Bij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3y = -Cij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4y = Dij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                
                DT_dx+=sup_i*(term1x+term2x+term3x+term4x)
                DT_dy+=sup_j*(term1y+term2y+term3y+term4y)

        self.grad_Fourier_from = DT_dx, DT_dy
        return DT_dx, DT_dy

    def get_grad_Fourier_formulation_vectorized(self) -> tuple:
        """
        Compute the derivative of the Fourier series signal using the optimized method.

        Returns:
        tuple: Derivatives of the Fourier series signal.
        """
        if self.print_state:
            print("\n\nFirst derivative of Fourier signal formulation (optimized method)...\n")

        DT_dx = np.zeros_like(self.filled_image, dtype=np.float64)
        DT_dy = np.zeros_like(self.filled_image, dtype=np.float64)

        # Precompute the sine and cosine terms
        hi = np.arange(self.harmonics_X)[:, np.newaxis, np.newaxis, np.newaxis]
        hj = np.arange(self.harmonics_Y)[np.newaxis, :, np.newaxis, np.newaxis]


        # Reshape grid_X and grid_Y to include an additional dimension for harmonics
        grid_X_reshaped = self.grid_X[np.newaxis, np.newaxis, :, :]
        grid_Y_reshaped = self.grid_Y[np.newaxis, np.newaxis, :, :]

        cos_X = np.cos(hi * grid_X_reshaped * np.pi / (self.Lx / 2))
        sin_X = np.sin(hi * grid_X_reshaped * np.pi / (self.Lx / 2))
        cos_Y = np.cos(hj * grid_Y_reshaped * np.pi / (self.Ly / 2))
        sin_Y = np.sin(hj * grid_Y_reshaped * np.pi / (self.Ly / 2))

        # Precompute the scaling factors
        sup_i = hi * np.pi / (self.Lx / 2)
        sup_j = hj * np.pi / (self.Ly / 2)

        # Extract Fourier coefficients
        Aij = self.Fourier_coefficients[0][:, :, np.newaxis, np.newaxis]
        Bij = self.Fourier_coefficients[1][:, :, np.newaxis, np.newaxis]
        Cij = self.Fourier_coefficients[2][:, :, np.newaxis, np.newaxis]
        Dij = self.Fourier_coefficients[3][:, :, np.newaxis, np.newaxis]

        # Compute the terms
        term1x = -Aij * sin_X * cos_Y
        term2x = -Bij * sin_X * sin_Y
        term3x = Cij * cos_X * cos_Y
        term4x = Dij * cos_X * sin_Y

        term1y = -Aij * cos_X * sin_Y
        term2y = Bij * cos_X * cos_Y
        term3y = -Cij * sin_X * sin_Y
        term4y = Dij * sin_X * cos_Y

        # Sum the terms
        DT_dx = np.sum(sup_i * (term1x + term2x + term3x + term4x), axis=(0, 1))
        DT_dy = np.sum(sup_j * (term1y + term2y + term3y + term4y), axis=(0, 1))

        self.grad_Fourier_from = DT_dx, DT_dy
        return DT_dx, DT_dy


    def get_regularized_grad(self, alpha: float = None) -> tuple:
        """
        Compute the regularized gradient according to the penalty factor alpha.

        Parameters:
        alpha (float): Regularization penalty factor. If None, use manual_alpha.

        Returns:
        tuple: Regularized gradient components.
        """
        if self.print_state:
            print("\n\nRegularized gradient computation...\n")

        if alpha is None:
            eff_alpha=self.manual_alpha
        else:
            eff_alpha=alpha


        f_x = np.zeros_like(self.filled_image, dtype=np.float64)
        f_y = np.zeros_like(self.filled_image, dtype=np.float64)

        for hi in tqdm(range(self.harmonics_X), desc="Processing harmonics X"):
            for hj in tqdm(range(self.harmonics_Y), desc="Processing harmonics Y", leave=False):

                Aij, Bij, Cij, Dij = self.Fourier_coefficients[:,hi,hj]
                Axij, Bxij, Cxij, Dxij, Ayij, Byij, Cyij, Dyij = self.solve_regularization_strong_equation(Aij, Bij, Cij, Dij, hi, hj, eff_alpha)
                        
                # F solution
                term1x = Axij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2x = Bxij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3x = Cxij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4x = Dxij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))

                term1y = Ayij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term2y = Byij*np.cos(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                term3y = Cyij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.cos(hj*self.grid_Y*np.pi/(self.Ly/2))
                term4y = Dyij*np.sin(hi*self.grid_X*np.pi/(self.Lx/2))*np.sin(hj*self.grid_Y*np.pi/(self.Ly/2))
                
                f_x+=term1x+term2x+term3x+term4x
                f_y+=term1y+term2y+term3y+term4y

        # Set the regularized value
        self.grad_regularized = f_x, f_y
        return f_x, f_y

    def get_regularized_grad_vectorized(self, alpha: float = None) -> tuple:
        """
        According to Mistral AI, based on 'get_regularized_grad'
        Compute the regularized gradient using a fully vectorized implementation.

        Parameters:
        alpha (float): Regularization penalty factor. If None, use self.manual_alpha.

        Returns:
        tuple: Regularized gradient components (f_x, f_y)
        """
        if self.print_state:
            print("\n\nRegularized gradient computation (vectorized)...\n")

        eff_alpha = self.manual_alpha if alpha is None else alpha

        f_x = np.zeros_like(self.filled_image, dtype=np.float64)
        f_y = np.zeros_like(self.filled_image, dtype=np.float64)

        # Precompute the sine and cosine terms
        hi = np.arange(self.harmonics_X)[:, np.newaxis, np.newaxis, np.newaxis]
        hj = np.arange(self.harmonics_Y)[np.newaxis, :, np.newaxis, np.newaxis]

        # Reshape grid_X and grid_Y to include additional dimensions for harmonics
        grid_X_reshaped = self.grid_X[np.newaxis, np.newaxis, :, :]
        grid_Y_reshaped = self.grid_Y[np.newaxis, np.newaxis, :, :]

        cos_X = np.cos(hi * grid_X_reshaped * np.pi / (self.Lx / 2))
        sin_X = np.sin(hi * grid_X_reshaped * np.pi / (self.Lx / 2))
        cos_Y = np.cos(hj * grid_Y_reshaped * np.pi / (self.Ly / 2))
        sin_Y = np.sin(hj * grid_Y_reshaped * np.pi / (self.Ly / 2))

        # Extract Fourier coefficients
        Aij = self.Fourier_coefficients[0]
        Bij = self.Fourier_coefficients[1]
        Cij = self.Fourier_coefficients[2]
        Dij = self.Fourier_coefficients[3]

        # Solve the regularization equation for all harmonics
        Axij_list = []
        Bxij_list = []
        Cxij_list = []
        Dxij_list = []
        Ayij_list = []
        Byij_list = []
        Cyij_list = []
        Dyij_list = []

        for hi in range(self.harmonics_X):
            for hj in range(self.harmonics_Y):
                Axij, Bxij, Cxij, Dxij, Ayij, Byij, Cyij, Dyij = self.solve_regularization_strong_equation(
                    Aij[hi, hj], Bij[hi, hj], Cij[hi, hj], Dij[hi, hj], hi, hj, eff_alpha
                )
                Axij_list.append(Axij)
                Bxij_list.append(Bxij)
                Cxij_list.append(Cxij)
                Dxij_list.append(Dxij)
                Ayij_list.append(Ayij)
                Byij_list.append(Byij)
                Cyij_list.append(Cyij)
                Dyij_list.append(Dyij)

        Axij = np.array(Axij_list).reshape((self.harmonics_X, self.harmonics_Y))
        Bxij = np.array(Bxij_list).reshape((self.harmonics_X, self.harmonics_Y))
        Cxij = np.array(Cxij_list).reshape((self.harmonics_X, self.harmonics_Y))
        Dxij = np.array(Dxij_list).reshape((self.harmonics_X, self.harmonics_Y))
        Ayij = np.array(Ayij_list).reshape((self.harmonics_X, self.harmonics_Y))
        Byij = np.array(Byij_list).reshape((self.harmonics_X, self.harmonics_Y))
        Cyij = np.array(Cyij_list).reshape((self.harmonics_X, self.harmonics_Y))
        Dyij = np.array(Dyij_list).reshape((self.harmonics_X, self.harmonics_Y))

        # Compute the terms
        term1x = Axij[:, :, np.newaxis, np.newaxis] * cos_X * cos_Y
        term2x = Bxij[:, :, np.newaxis, np.newaxis] * cos_X * sin_Y
        term3x = Cxij[:, :, np.newaxis, np.newaxis] * sin_X * cos_Y
        term4x = Dxij[:, :, np.newaxis, np.newaxis] * sin_X * sin_Y

        term1y = Ayij[:, :, np.newaxis, np.newaxis] * cos_X * cos_Y
        term2y = Byij[:, :, np.newaxis, np.newaxis] * cos_X * sin_Y
        term3y = Cyij[:, :, np.newaxis, np.newaxis] * sin_X * cos_Y
        term4y = Dyij[:, :, np.newaxis, np.newaxis] * sin_X * sin_Y

        # Sum the terms
        f_x = np.sum(term1x + term2x + term3x + term4x, axis=(0, 1))
        f_y = np.sum(term1y + term2y + term3y + term4y, axis=(0, 1))

        self.grad_regularized = f_x, f_y
        
        return f_x, f_y

    
    def get_thermal_gradient_norm_and_orientation(self) -> tuple:
        """
        Compute the norm and orientation of the thermal gradient vectors.

        The norms are in units of K/px (Kelvin per pixel).
        The angles are in radians between the positive x-axis and the vector defined by the gradient components.

        Returns:
        tuple: A tuple containing the norm and the orientation of the thermal gradient vectors.
        """
        # Ensure grad_regularized is computed
        if self.grad_regularized is None:
            raise ValueError("The regularized gradient has not been computed yet. Please run the regularization method first.")

        # Extract the gradient components
        grad_x, grad_y = self.grad_regularized

        # Compute the norm of the gradient
        gradient_norm = np.sqrt(grad_x**2 + grad_y**2)

        # Compute the orientation of the gradient vectors
        gradient_orientation = np.arctan2(grad_y, grad_x)

        return gradient_norm, gradient_orientation


    # Visualization
    def show_raw_data(self):
        """ Show raw image. """

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        data=self.raw_image
        cax = ax.imshow(data)

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("bottom", size=0.2, pad=0.5)
        cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Température [°C]")

        plt.show()

    def show_raw_data_extended(self):
        """ Show raw image. """
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        data=np.ma.masked_array(self.filled_image, mask=self.mask_filled)
        cax = ax.imshow(data)

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("bottom", size=0.2, pad=0.5)
        cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Température [°C]")

        plt.show()

    def show_filled_data_extended(self):
        """ """
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        data=self.filled_image
        cax = ax.imshow(data)

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("bottom", size=0.2, pad=0.5)
        cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Température [°C]")

        plt.show()

    def show_Fourier_form(self):
        """ """
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        data=self.get_Fourier_formulation_vectorized()
        cax = ax.imshow(data)

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("bottom", size=0.2, pad=0.5)
        cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Température [°C]")

        plt.show()   

    def show_profil_evolution(self, line:int, col:int):

        data1=np.ma.masked_array(self.filled_image, mask=self.mask_filled)
        data2=np.ma.masked_array(self.get_Fourier_formulation_vectorized(), mask=self.mask_filled)
        data3_x=np.ma.masked_array(self.grad_Fourier_from[0], mask=self.mask_filled)
        data3_y=np.ma.masked_array(self.grad_Fourier_from[1], mask=self.mask_filled)
        data4_x=np.ma.masked_array(self.grad_regularized[0], mask=self.mask_filled)
        data4_y=np.ma.masked_array(self.grad_regularized[1], mask=self.mask_filled)
        
        with plt.ion():
            # Comparaison des deux images
            plt.figure("Visualisation des lignes de profil")
            plt.clf()  

            ## Brutes
            plt.subplot(1, 2, 1)
            plt.imshow(data1)
            plt.hlines(y=line, xmin=0, xmax=data1.shape[1], color='b')
            plt.vlines(x=col, ymin=0, ymax=data1.shape[0], color='b')
            plt.xlim(0, data1.shape[1])
            plt.ylim(data1.shape[0], 0)

            ## Fourier
            plt.subplot(1, 2, 2)
            plt.imshow(data2)
            plt.hlines(y=line, xmin=0, xmax=data2.shape[1], color='r')
            plt.vlines(x=col, ymin=0, ymax=data2.shape[0], color='r')
            plt.xlim(0, data2.shape[1])
            plt.ylim(data2.shape[0], 0)

            plt.show()

            # Profil lines
            plt.figure("Profil")
            plt.clf()

            plt.subplot(2, 2, 1)
            plt.title(f"Line n°{line}")
            plt.plot(data1[line,:], c='b', label='Raw data')
            plt.plot(data2[line,:], c='r', label='Fourier signal')
            plt.legend()
            
            plt.subplot(2, 2, 3)
            plt.plot(data4_x[line,:], c='k', label='Regularized gradient')
            plt.plot(data3_x[line,:], 'k--', label='First derivative Fourier signal')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.title(f"Columns n°{col}")
            plt.plot(data1[:,col], c='b', label='Raw data')
            plt.plot(data2[:,col], c='r', label='Fourier signal')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.plot(data4_y[:,col], c='k', label='Regularized gradient')
            plt.plot(data3_y[:,col], 'k--', label='First derivative Fourier signal')

            plt.legend()
            plt.show()

    def show_L_curve(self, alpha_range=np.arange(0.34, 0.36, 0.001)):

        best_alpha, L, J = self.L_curve(alpha_range=alpha_range)
        id_ = np.where(best_alpha == alpha_range)[0][0]

        plt.plot(L, J, 'ko--')
        plt.scatter(L[0], J[0], color='b', s=125, label=fr"$\alpha = {alpha_range[0]:.3f}$")
        plt.scatter(L[id_], J[id_], color='r', s=125, label=fr"$\alpha = {best_alpha:.3f}$")
        plt.scatter(L[-1], J[-1], color='g', s=125, label=fr"$\alpha = {alpha_range[-1]:.3f}$")
        plt.xlabel(r"$L(f_\alpha)$")
        plt.ylabel(r"$J(f_\alpha)$")
        plt.legend()
        plt.show()

    def show_gradients(self, T_target):

        regul_fx, regul_fy = self.grad_regularized
        gradient_norm = np.sqrt(regul_fx**2 + regul_fy**2)
        gradient_angle = np.arctan2(regul_fx, regul_fy)
        image_thermique=self.get_Fourier_formulation_vectorized()
        mask_image_thermique = np.ma.array(image_thermique, mask=self.mask_filled)

        # Les angles vont de -pi à pi, l'angle 0 correspond à la verticale descendante.
        # Prendre les angles aux valeurs suppérieures à pi/2 pointent en haut à droite.
        mask_gradient = (
              (image_thermique > T_target-5) & (image_thermique < T_target+5) \
            & ((gradient_angle > np.pi/2) | (gradient_angle < -np.pi/2))
        )
        
        # Appliquer le masque
        X, Y = np.meshgrid(
            np.arange(image_thermique.shape[1]), 
            np.arange(image_thermique.shape[0])
            )
        masked_X = X[mask_gradient == 1]
        masked_Y = Y[mask_gradient == 1]

        masked_gradient_x = regul_fx[mask_gradient == 1]
        masked_gradient_y = regul_fy[mask_gradient == 1]
        masked_gradient_norm = gradient_norm[mask_gradient == 1] 
    
        # Affichage
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(image_thermique, cmap='viridis')
        
        quiver = ax.quiver(
            masked_X, masked_Y, 
            masked_gradient_x, masked_gradient_y, 
            masked_gradient_norm, 
            angles='xy', scale_units='xy', scale=5, cmap='coolwarm'
            )

        fig.colorbar(cax, ax=ax, orientation='horizontal', label='T [°C]')
        fig.colorbar(quiver, ax=ax, orientation='horizontal', label=r'$|\Delta T|$  [°C/mm]')
        plt.title('Carte thermique et gradient thermique autour de '+ r'$T_{objectif}$' +f' = {T_target}°C')
        plt.show()


def convergence_analyse(img: np.ndarray, harmonic_to_test = np.arange(10, 61, 5)) -> None:
    """
    Analyze the convergence of the Fourier series representation by varying the number of harmonics.

    Args:
        img (np.ndarray): The input thermal image.

    Returns:
        None
    """
    data = {}
    
    for Nx in harmonic_to_test:
        for Ny in harmonic_to_test:
            print(f"\n\n\n{Nx} {Ny}\n********")
            GradThermique = RegularizedThermalGradient(
                raw_image=img,
                crop_data=(200, 0, 200, 639),
                harmonics_X=Nx,
                harmonics_Y=Ny,
                scale=1,
                manual_alpha=0.35,
                optimized=True
            )
            data[f"{Nx}_{Ny}"] = GradThermique

    error = np.zeros(shape=(harmonic_to_test.size, harmonic_to_test.size))
    for ix, Nx in enumerate(harmonic_to_test):
        for iy, Ny in enumerate(harmonic_to_test):
            Four = np.ma.array(
                data[f"{Nx}_{Ny}"].Fourier_form,
                mask=data[f"{Nx}_{Ny}"].mask_filled
            )
            Ref = np.ma.array(
                data[f"{Nx}_{Ny}"].filled_image,
                mask=data[f"{Nx}_{Ny}"].mask_filled
            )
            error[ix, iy] = np.abs(Four.data - Ref.data).mean()

    XNx, YNy = np.meshgrid(harmonic_to_test, harmonic_to_test)

    fig, ax = plt.subplots()
    sc = ax.scatter(XNx, YNy, c=error, cmap='viridis', marker="x")
    plt.colorbar(sc, label=r'Error ($\xi$)')

    # Adding titles and labels
    ax.set_xlabel(r"Number of harmonics ($N_x$)")
    ax.set_ylabel(r"Number of harmonics ($N_y$)")

    # Adding grid
    ax.grid(True, linestyle='--', linewidth=0.5)

    # Removing the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()
