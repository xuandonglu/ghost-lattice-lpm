import numpy as np

R = 0.5


def material(lattice, E, nu, r, plane="stress"):
    if lattice == "hex":
        if plane == "stress":
            C11 = E / (1 - nu**2)
            C12 = nu * E / (1 - nu**2)
        else:
            C11 = E * (1 - nu) / (1 + nu) / (1 - 2 * nu)
            C12 = E * nu / (1 + nu) / (1 - 2 * nu)
        Kn = np.sqrt(3) / 12 * C11 - np.sqrt(3) / 12 * C12
        T = -np.sqrt(3) / 144 * C11 + np.sqrt(3) / 48 * C12
        return dict(
            Kn1=Kn,
            Kn2=Kn,
            T=T,
            V=2 * np.sqrt(3) * r**2 * 1.0,
            C11=C11,
            C12=C12,
            C44=(C11 - C12) / 2,
        )
    if lattice == "sc":
        C11 = E * (1 - nu) / (1 + nu) / (1 - 2 * nu)
        C12 = E * nu / (1 + nu) / (1 - 2 * nu)
        C44 = E / 2 / (1 + nu)
        return dict(
            Kn1=r * (C11 - C12 - C44),
            Kn2=r * C44,
            T=r * (C12 - C44) / 18.0,
            V=(2 * r) ** 3,
            C11=C11,
            C12=C12,
            C44=C44,
        )
    raise ValueError


def mat2d(E=1.0, nu=0.3):
    return material("hex", E, nu, R, "stress")


def Cmat(E=1.0, nu=0.3):
    c = E / (1 - nu**2)
    return np.array([[c, c * nu, 0], [c * nu, c, 0], [0, 0, E / 2 / (1 + nu)]])
