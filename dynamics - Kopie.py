"""
defines the parameter and dynamics of the pendubot.

Run derive_dynamics.py before use to create text files that contain the dynamics.
Be careful this script uses eval()!!! Thus it interprets some text written in a file as code.

functions:
- get_dxdt gets() returns the rhs of the non-linear dynamics
- get_dxdt_casadi() returns a casadi model of the non-linear dynamics
- get_linear_dynamics() returns the A,b of the linearized dynamics
"""
try:
    with open('dynamics_phi_dd.txt', 'r') as f:
        ddphi_expr = f.read()
    with open('dynamics_x_dd.txt', 'r') as f:
        dds_expr = f.read()
    with open('dynamics_linearized_A.txt', 'r') as f:
        A_expr = f.read()
    with open('dynamics_linearized_b.txt', 'r') as f:
        b_expr = f.read()
except FileNotFoundError:
    raise FileNotFoundError('dynamic equations not found. Run derive_dynamics.py first.')

DEFAULT_PARAMETER = {
    'g': 9.81,
    'M': 0.7,
    'm': 0.221,
    'l': 0.5,
    'b': 0.02,
    'c': 1
}


def _unpack_parameter(parameter):
    g = parameter['g']
    m = parameter['m']
    M = parameter['M']
    l = parameter['l']
    b = parameter['b']
    c = parameter['c']

    return g, m, M, l, b, c


def get_dxdt(parameter=None):
    """
    gets function for right side of the actuated double pendulum: d/dt x = f(t, x, voltage).

    The state x is given by (phi1, dphi1, phi2, dphi2).

    Parameters
    ----------
    parameter : Dict
        pendulum parameter

    Returns
    -------
    Callable[[float, np.ndarray, float], [np.ndarray]]
        f(t, x, voltage)
    """
    import numpy as np
    if parameter is None:
        parameter = DEFAULT_PARAMETER

    def dx_dt(t, x, parameter, F):
        phi, dphi, s, ds = x
        # needed for eval of txt file
        from numpy import sin, cos
        g, m, M, l, b, c  = _unpack_parameter(parameter)
        return np.array([
            dphi,
            eval(ddphi_expr),
            ds,
            eval(dds_expr),
        ])

    return dx_dt


def get_dxdt_casadi(parameter):
    """
    gets casadi function for right side of the actuated double pendulum: d/dt x = f(t, x, voltage).

    The state x is given by (phi1, dphi1, phi2, dphi2).

    Parameters
    ----------
    parameter : Dict
        pendulum parameter

    Returns
    -------
    Tuple[ca.Function, ca.SX, ca.SX]
        f(t, x, voltage), x, voltage
    """
    import casadi as ca
    if parameter is None:
        parameter = DEFAULT_PARAMETER

    # needed for eval of txt file
    g, m, M, l, b, c  = _unpack_parameter(parameter)
    from casadi import sin, cos

    x = ca.SX.sym('x', (4, 1))
    t = ca.SX.sym('t')
    phi, dphi, s, ds = ca.vertsplit(x)
    F = ca.SX.sym('F')

    dx = ca.vertcat(
        dphi,
        eval(ddphi_expr),
        ds,
        eval(dds_expr)
    )

    f = ca.Function('f', [t, x, F], [dx])

    return f, x, F


def get_linear_dynamics(t, x, voltage, parameter=None):
    """
    gets the linearized dynamics for a given time, state and input.

    Parameters
    ----------
    t: float
        time
    x: np.ndarray
        state
    voltage: float
        input
    parameter: Dict (default None)
        parameter of the pendubot. Uses DEFAULT_PARAMETER if None.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        System matrix A (4x4), input vector b (4x1)

    """
    if parameter is None:
        parameter = DEFAULT_PARAMETER

    g, m, M, l, b, c = _unpack_parameter(parameter)
    phi, dphi, s, ds = x
    import numpy as np
    from casadi import sin, cos
    A = np.array(eval(A_expr))
    b = np.array(eval(b_expr))
    return A.transpose(), b.transpose()


def _test_lin():
    import numpy as np

    x = np.array([1, 2, 1.1, 2.1])
    u = np.array(12)
    dxdt = get_dxdt()
    df_du_approx = ((dxdt(0, x, u + 0.0000000001) - dxdt(0, x, u)) / 0.0000000001).reshape((4, 1))
    df_dx, df_du = get_linear_dynamics(0, x, u)

    print((df_du_approx - df_du).max())

    df_dx_approx = np.vstack(
        [(dxdt(0, x + 0.000000001 * e, u) - dxdt(0, x, u)) / 0.000000001 for e in
         np.eye(4)]).transpose()

    print((df_dx_approx - df_dx).max())