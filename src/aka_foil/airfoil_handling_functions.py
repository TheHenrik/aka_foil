""" 
Functions to handle airfoil data. Some code from neuralfoil.
"""

import numpy as np
from typing import List, Optional, Dict, Union
from scipy.special import comb
import re
import os
from pathlib import Path

def get_kulfan_parameters(
    coordinates: np.ndarray,
    n_weights_per_side: int = 8,
    N1: float = 0.5,
    N2: float = 1.0,
    n_points_per_side: int = 200,
    normalize_coordinates: bool = True,
    use_leading_edge_modification: bool = True,
    method: str = "least_squares",
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Given a set of airfoil coordinates, reconstructs the Kulfan parameters that would recreate that airfoil. Uses a
    curve fitting (optimization) process.

    This function is the inverse of `get_kulfan_coordinates()`.

    Kulfan parameters are a highly-efficient and flexible way to parameterize the shape of an airfoil. The particular
    flavor of Kulfan parameterization used in AeroSandbox is the "CST with LEM" method, which is described in various
    papers linked below. In total, the Kulfan parameterization consists of:

    * A vector of weights corresponding to the lower surface of the airfoil
    * A vector of weights corresponding to the upper surface of the airfoil
    * A scalar weight corresponding to the strength of a leading-edge camber mode shape of the airfoil (optional)
    * The trailing-edge (TE) thickness of the airfoil (optional)

    These Kulfan parameters are also referred to as CST (Class/Shape Transformation) parameters.

    References on Kulfan (CST) airfoils:

    * Kulfan, Brenda "Universal Parametric Geometry Representation Method" (2008). AIAA Journal of Aircraft.
        Describes the basic Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.29958
            * https://www.brendakulfan.com/_files/ugd/169bff_6738e0f8d9074610942c53dfaea8e30c.pdf
            * https://www.researchgate.net/publication/245430684_Universal_Parametric_Geometry_Representation_Method

    * Kulfan, Brenda "Modification of CST Airfoil Representation Methodology" (2020). Unpublished note:
        Describes the optional "Leading-Edge Modification" (LEM) addition to the Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://www.brendakulfan.com/_files/ugd/169bff_16a868ad06af4fea946d299c6028fb13.pdf
            * https://www.researchgate.net/publication/343615711_Modification_of_CST_Airfoil_Representation_Methodology

    * Masters, D.A. "Geometric Comparison of Aerofoil Shape Parameterization Methods" (2017). AIAA Journal.
        Compares the Kulfan (CST) airfoil parameterization to other airfoil parameterizations. Also has further notes
        on the LEM addition.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.J054943
            * https://research-information.bris.ac.uk/ws/portalfiles/portal/91793513/SP_Journal_RED.pdf

    Notes on N1, N2 (shape factor) combinations:
        * 0.5, 1: Conventional airfoil
        * 0.5, 0.5: Elliptic airfoil
        * 1, 1: Biconvex airfoil
        * 0.75, 0.75: Sears-Haack body (radius distribution)
        * 0.75, 0.25: Low-drag projectile
        * 1, 0.001: Cone or wedge airfoil
        * 0.001, 0.001: Rectangle, circular duct, or circular rod.

    The following demonstrates the reversibility of this function:

    Args:

        coordinates (np.ndarray): The coordinates of the airfoil as a Nx2 array.

        n_weights_per_side (int): The number of Kulfan weights to use per side of the airfoil.

        N1 (float): The shape factor corresponding to the leading edge of the airfoil. See above for examples.

        N2 (float): The shape factor corresponding to the trailing edge of the airfoil. See above for examples.

        n_points_per_side (int): The number of points to discretize with, when formulating the curve-fitting
            optimization problem.

    Returns:
        A dictionary containing the Kulfan parameters. The keys are:
            * "lower_weights" (np.ndarray): The weights corresponding to the lower surface of the airfoil.
            * "upper_weights" (np.ndarray): The weights corresponding to the upper surface of the airfoil.
            * "TE_thickness" (float): The trailing-edge thickness of the airfoil.
            * "leading_edge_weight" (float): The strength of the leading-edge camber mode shape of the airfoil.

        These can be passed directly into `get_kulfan_coordinates()` to reconstruct the airfoil.
    """

    if method == "least_squares":

        """

        The goal here is to set up this fitting problem as a least-squares problem (likely an overconstrained one,
        but keeping it general for now. This will then be solved with np.linalg.lstsq(A, b), where A will (likely)
        not be square.

        The columns of the A matrix will correspond to our unknowns, which are going to be a 1D vector `x` packed in as:
            * upper_weights from 0 to n_weights_per_side - 1
            * lower_weights from 0 to n_weights_per_side - 1
            * leading_edge_weight
            * trailing_edge_thickness

        See `get_kulfan_coordinates()` for more details on the meaning of these variables.

        The rows of the A matrix will correspond to each row of the given airfoil coordinates (i.e., a single vertex
        on the airfoil). The idea here is to express each vertex as a linear combination of the unknowns, and then
        solve for the unknowns that minimize the error between the given airfoil coordinates and the reconstructed
        airfoil coordinates.

        """

        n_coordinates = len(coordinates)

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        LE_index = np.argmin(x)
        is_upper = np.arange(len(x)) <= LE_index

        # Class function
        C = (x) ** N1 * (1 - x) ** N2

        # Shape function (Bernstein polynomials)
        N = n_weights_per_side - 1  # Order of Bernstein polynomials

        K = comb(N, np.arange(N + 1))  # Bernstein polynomial coefficients

        dims = (n_weights_per_side, n_coordinates)

        def wide(vector):
            return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

        def tall(vector):
            return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

        S_matrix = (
            tall(K)
            * wide(x) ** tall(np.arange(N + 1))
            * wide(1 - x) ** tall(N - np.arange(N + 1))
        )  # Bernstein polynomial coefficients * weight matrix

        leading_edge_weight_row = x * np.maximum(1 - x, 0) ** (n_weights_per_side + 0.5)

        trailing_edge_thickness_row = np.where(is_upper, x / 2, -x / 2)

        A = np.concatenate(
            [
                np.where(wide(is_upper), 0, wide(C) * S_matrix).T,
                np.where(wide(is_upper), wide(C) * S_matrix, 0).T,
                np.reshape(leading_edge_weight_row, (n_coordinates, 1)),
                np.reshape(trailing_edge_thickness_row, (n_coordinates, 1)),
            ],
            axis=1,
        )

        b = y

        # Solve least-squares problem
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        lower_weights = x[:n_weights_per_side]
        upper_weights = x[n_weights_per_side : 2 * n_weights_per_side]
        leading_edge_weight = x[-2]
        trailing_edge_thickness = x[-1]

        # If you got a negative trailing-edge thickness, then resolve the problem with a TE_thickness = 0 constraint.
        if trailing_edge_thickness < 0:

            x, _, _, _ = np.linalg.lstsq(A[:, :-1], b, rcond=None)

            lower_weights = x[:n_weights_per_side]
            upper_weights = x[n_weights_per_side : 2 * n_weights_per_side]
            leading_edge_weight = x[-1]
            trailing_edge_thickness = 0

        return {
            "lower_weights": lower_weights,
            "upper_weights": upper_weights,
            "TE_thickness": trailing_edge_thickness,
            "leading_edge_weight": leading_edge_weight,
        }

    else:
        raise ValueError(f"Invalid method '{method}'.")

def get_coordinates_from_raw_dat(raw_text: List[str]) -> np.ndarray:
    """
    Returns a Nx2 ndarray of airfoil coordinates from the raw text of a airfoil *.dat file.

    Args:

        raw_text: A list of strings, where each string is one line of the *.dat file. One good way to get this input
            is to read the file via the `with open(file, "r") as file:`, `file.readlines()` interface.

    Returns: A Nx2 ndarray of airfoil coordinates [x, y].

    """
    raw_coordinates = []

    def is_number(s: str) -> bool:
        # Determines whether a string is representable as a float
        try:
            float(s)
        except ValueError:
            return False
        return True

    def parse_line(line: str) -> Optional[List[float]]:
        # Given a single line of a `*.dat` file, tries to parse it into a list of two floats [x, y].
        # If not possible, returns None.
        line_split = re.split(r"[;|,|\s|\t]", line)
        line_items = [s for s in line_split if s != ""]
        if len(line_items) == 2 and all([is_number(item) for item in line_items]):
            return line_items
        else:
            return None

    for line in raw_text:
        parsed_line = parse_line(line)
        if parsed_line is not None:
            raw_coordinates.append(parsed_line)

    if len(raw_coordinates) == 0:
        raise ValueError("Could not read any coordinates from the `raw_text` input!")

    coordinates = np.array(raw_coordinates, dtype=float)

    return coordinates

def get_file_coordinates(filepath: Union[str, os.PathLike]):
    possible_errors = (FileNotFoundError, UnicodeDecodeError)

    if isinstance(filepath, np.ndarray):
        raise TypeError("`filepath` should be a string or os.PathLike object.")

    try:
        with open(filepath, "r") as f:
            raw_text = f.readlines()
    except possible_errors:
        try:
            with open(f"{filepath}.dat", "r") as f:
                raw_text = f.readlines()
        except possible_errors as e:
            raise FileNotFoundError(
                f" Neither '{filepath}' nor '{filepath}.dat' were found and readable."
            ) from e

    try:
        return get_coordinates_from_raw_dat(raw_text)
    except ValueError:
        raise ValueError("File was found, but could not read any coordinates!")

def get_kulfan_coordinates(
    lower_weights: np.ndarray = -0.2 * np.ones(8),
    upper_weights: np.ndarray = 0.2 * np.ones(8),
    leading_edge_weight: float = 0.0,
    TE_thickness: float = 0.0,
    n_points_per_side: int = 200,
    N1: float = 0.5,
    N2: float = 1.0,
    **deprecated_kwargs,
) -> np.ndarray:
    """
    Given a set of Kulfan parameters, computes the coordinates of the resulting airfoil.

    This function is the inverse of `get_kulfan_parameters()`.

    Kulfan parameters are a highly-efficient and flexible way to parameterize the shape of an airfoil. The particular
    flavor of Kulfan parameterization used in AeroSandbox is the "CST with LEM" method, which is described in various
    papers linked below. In total, the Kulfan parameterization consists of:

    * A vector of weights corresponding to the lower surface of the airfoil
    * A vector of weights corresponding to the upper surface of the airfoil
    * A scalar weight corresponding to the strength of a leading-edge camber mode shape of the airfoil (optional)
    * The trailing-edge (TE) thickness of the airfoil (optional)

    These Kulfan parameters are also referred to as CST (Class/Shape Transformation) parameters.

    References on Kulfan (CST) airfoils:

    * Kulfan, Brenda "Universal Parametric Geometry Representation Method" (2008). AIAA Journal of Aircraft.
        Describes the basic Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.29958
            * https://www.brendakulfan.com/_files/ugd/169bff_6738e0f8d9074610942c53dfaea8e30c.pdf
            * https://www.researchgate.net/publication/245430684_Universal_Parametric_Geometry_Representation_Method

    * Kulfan, Brenda "Modification of CST Airfoil Representation Methodology" (2020). Unpublished note:
        Describes the optional "Leading-Edge Modification" (LEM) addition to the Kulfan (CST) airfoil parameterization.
        Mirrors:
            * https://www.brendakulfan.com/_files/ugd/169bff_16a868ad06af4fea946d299c6028fb13.pdf
            * https://www.researchgate.net/publication/343615711_Modification_of_CST_Airfoil_Representation_Methodology

    * Masters, D.A. "Geometric Comparison of Aerofoil Shape Parameterization Methods" (2017). AIAA Journal.
        Compares the Kulfan (CST) airfoil parameterization to other airfoil parameterizations. Also has further notes
        on the LEM addition.
        Mirrors:
            * https://arc.aiaa.org/doi/10.2514/1.J054943
            * https://research-information.bris.ac.uk/ws/portalfiles/portal/91793513/SP_Journal_RED.pdf

    Notes on N1, N2 (shape factor) combinations:
        * 0.5, 1: Conventional airfoil
        * 0.5, 0.5: Elliptic airfoil
        * 1, 1: Biconvex airfoil
        * 0.75, 0.75: Sears-Haack body (radius distribution)
        * 0.75, 0.25: Low-drag projectile
        * 1, 0.001: Cone or wedge airfoil
        * 0.001, 0.001: Rectangle, circular duct, or circular rod.

    To make a Kulfan (CST) airfoil, use the following syntax:


    Args:

        lower_weights (iterable): The Kulfan weights to use for the lower surface.

        upper_weights (iterable): The Kulfan weights to use for the upper surface.

        TE_thickness (float): The trailing-edge thickness to add, in terms of y/c.

        n_points_per_side (int): The number of points to discretize with, when generating the coordinates.

        N1 (float): The shape factor corresponding to the leading edge of the airfoil. See above for examples.

        N2 (float): The shape factor corresponding to the trailing edge of the airfoil. See above for examples.

    Returns:
        np.ndarray: The coordinates of the airfoil as a Nx2 array.
    """
    if len(deprecated_kwargs) > 0:
        import warnings

        warnings.warn(
            "The following arguments are deprecated and will be removed in a future version:\n"
            f"{deprecated_kwargs}",
            DeprecationWarning,
        )

        if deprecated_kwargs.get("enforce_continuous_LE_radius", False):
            lower_weights[0] = -1 * upper_weights[0]

    def cosine_spaced_points(start, end, num_points):
        # Erzeuge gleichmäßig verteilte Winkel im Bereich [0, π]
        theta = np.linspace(0, np.pi, num_points)
        # Wende die Cosinus-Transformation an und skaliere auf den Bereich [start, end]
        points = (start + end) / 2 + (end - start) / 2 * np.cos(theta)
        return points

    x = cosine_spaced_points(0, 1, n_points_per_side)  # Generate some cosine-spaced points

    # Class function
    C = (x) ** N1 * (1 - x) ** N2

    def shape_function(w):
        # Shape function (Bernstein polynomials)
        N = len(w) - 1  # Order of Bernstein polynomials

        K = comb(N, np.arange(N + 1))  # Bernstein polynomial coefficients

        dims = (len(w), len(x))

        def wide(vector):
            return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

        def tall(vector):
            return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

        S_matrix = (
            tall(K)
            * wide(x) ** tall(np.arange(N + 1))
            * wide(1 - x) ** tall(N - np.arange(N + 1))
        )  # Bernstein polynomial coefficients * weight matrix
        S_x = np.sum(tall(w) * S_matrix, axis=0)

        # Calculate y output
        y = C * S_x
        return y

    y_lower = shape_function(lower_weights)
    y_upper = shape_function(upper_weights)

    # Add trailing-edge (TE) thickness
    y_lower -= x * TE_thickness / 2
    y_upper += x * TE_thickness / 2

    # Add Kulfan's leading-edge-modification (LEM)
    y_lower += leading_edge_weight * (x) * (1 - x) ** (len(lower_weights) + 0.5)
    y_upper += leading_edge_weight * (x) * (1 - x) ** (len(upper_weights) + 0.5)

    x = np.concatenate((x[::-1], x[1:]))
    y = np.concatenate((y_upper[::-1], y_lower[1:]))
    coordinates = np.stack((x, y), axis=1)

    return coordinates

def get_data_from_xfoil(foil_name: str, re: float, num_points: int = 100):
    airfoil_results_path = os.path.join(
        Path(__file__).parent.parent.parent, "data", "xfoil_results", f"{foil_name}.csv"
    )
    polar_data = np.loadtxt(airfoil_results_path, delimiter=",", skiprows=1)

    re_list = np.unique(polar_data[:, 0])

    if re > re_list[-1]:
        print(
            "Warning: Airfoil: %s -> Re=%.0f above max Re in surrogate model"
            % (foil_name, re)
        )
        re = re_list[-1]
    upper_re = re_list[np.where(re_list >= re)[0][0]]
    if np.where(re_list >= re)[0][0] == 0:
        lower_re = re_list[np.where(re_list >= re)[0][0]]
        print(
            "Warning: Airfoil: %s -> Re=%.0f below min Re in surrogate model"
            % (foil_name, re)
        )
    else:
        lower_re = re_list[np.where(re_list >= re)[0][0] - 1]

    polar_data_upper = polar_data[np.where(polar_data[:, 0] == upper_re)[0], :]
    polar_data_lower = polar_data[np.where(polar_data[:, 0] == lower_re)[0], :]

    min_cl = max(np.min(polar_data_upper[:, 2]), np.min(polar_data_lower[:, 2]))
    max_cl = min(np.max(polar_data_upper[:, 2]), np.max(polar_data_lower[:, 2]))
    cls = np.linspace(min_cl, max_cl, num_points)
    cds = np.array([])
    cms = np.array([])
    top_xtrs = np.array([])
    bot_xtrs = np.array([])
    alphas = np.array([])

    for cl in cls:
        CD_upper = np.interp(
            cl, polar_data_upper[:, 2], polar_data_upper[:, 3], left=1.0, right=1.0
        )
        CD_lower = np.interp(
            cl, polar_data_lower[:, 2], polar_data_lower[:, 3], left=1.0, right=1.0
        )
        CD = np.interp(re, [lower_re, upper_re], [CD_lower, CD_upper])
        cds = np.append(cds, CD)

        CM_upper = np.interp(
            cl, polar_data_upper[:, 2], polar_data_upper[:, 5], left=1.0, right=1.0
        )
        CM_lower = np.interp(
            cl, polar_data_lower[:, 2], polar_data_lower[:, 5], left=1.0, right=1.0
        )
        CM = np.interp(re, [lower_re, upper_re], [CM_lower, CM_upper])
        cms = np.append(cms, CM)

        top_xtr_upper = np.interp(
            cl, polar_data_upper[:, 2], polar_data_upper[:, 6], left=1.0, right=1.0
        )
        top_xtr_lower = np.interp(
            cl, polar_data_lower[:, 2], polar_data_lower[:, 6], left=1.0, right=1.0
        )
        top_xtr = np.interp(re, [lower_re, upper_re], [top_xtr_lower, top_xtr_upper])
        top_xtrs = np.append(top_xtrs, top_xtr)

        bot_xtr_upper = np.interp(
            cl, polar_data_upper[:, 2], polar_data_upper[:, 7], left=1.0, right=1.0
        )
        bot_xtr_lower = np.interp(
            cl, polar_data_lower[:, 2], polar_data_lower[:, 7], left=1.0, right=1.0
        )
        bot_xtr = np.interp(re, [lower_re, upper_re], [bot_xtr_lower, bot_xtr_upper])
        bot_xtrs = np.append(bot_xtrs, bot_xtr)

        alpha_upper = np.interp(
            cl, polar_data_upper[:, 2], polar_data_upper[:, 1], left=1.0, right=1.0
        )
        alpha_lower = np.interp(
            cl, polar_data_lower[:, 2], polar_data_lower[:, 1], left=1.0, right=1.0
        )
        alpha = np.interp(re, [lower_re, upper_re], [alpha_lower, alpha_upper])
        alphas = np.append(alphas, alpha)

    return np.stack([alphas, cls, cds, cms, top_xtrs, bot_xtrs], axis=1)


if __name__ == '__main__':
    coordinates = get_file_coordinates('../../data/airfoils/acc22.dat')

    kulfan_parameters = get_kulfan_parameters(coordinates, n_weights_per_side=2)
    coordinates_reconstructed_low = get_kulfan_coordinates(**kulfan_parameters)

    kulfan_parameters = get_kulfan_parameters(coordinates, n_weights_per_side=8)
    coordinates_reconstructed_high = get_kulfan_coordinates(**kulfan_parameters)

    coordinates = np.array(coordinates)
    coordinates_reconstructed_low = np.array(coordinates_reconstructed_low)

    coordinates = np.array(coordinates)
    coordinates_reconstructed_high = np.array(coordinates_reconstructed_high)

    import matplotlib.pyplot as plt
    plt.scatter(coordinates[:, 0], coordinates[:, 1], label='Original', marker='x', color='red')
    plt.plot(coordinates_reconstructed_low[:, 0], coordinates_reconstructed_low[:, 1], label='2 kulfan weights per side')
    plt.plot(coordinates_reconstructed_high[:, 0], coordinates_reconstructed_high[:, 1], label='8 kulfan weights per side')
    plt.legend(fontsize=14, loc='lower right')
    plt.savefig('kulfan_parameters.pdf')
    plt.show()
    # import matplotlib.pyplot as plt
    # plt.plot(get_data_from_xfoil('ag45c', 400_000, num_points=10)[:, 0], get_data_from_xfoil('ag45c', 400_000, num_points=10)[:, 4])
    # plt.show()