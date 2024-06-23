"""Toy code implementing a matrix product state."""

import numpy as np
from scipy.linalg import svd

class MPS:
    """Class for a matrix product state.

    We index sites with `i` from 0 to L-1; bond `i` is left of site `i`.
    We assume that the state is in right-canonical form.

    Parameters
    ----------
    Bs, Ss:
        Same as attributes.

    Attributes
    ----------
    Bs : list of np.Array[ndim=3]
        The 'matrices' in right-canonical form, one for each physical site.
        Each `B[i]` has legs (virtual left, physical, virtual right), in short ``vL i vR``
    Ss : list of np.Array[ndim=1]
        The Schmidt values at each of the bonds, ``Ss[i]`` is left of ``Bs[i]``.
    L : int
        Number of sites.
    """

    def __init__(self, Bs, Ss):
        self.Bs = Bs
        self.Ss = Ss
        self.L = len(Bs)

    def copy(self):
        return MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss])

    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs)."""
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``."""
        j = i + 1
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.L - 1)]

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op, theta, axes=[1, 1])  # i [i*], vL [i] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.L - 1):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = np.tensordot(op[i], theta, axes=[[2, 3], [1, 2]])
            # i j [i*] [j*], vL [i] [j] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(result)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        result = []
        for i in range(1, self.L):
            S = self.Ss[i].copy()
            S = S[S > 1e-30]  # 0*log(0) should give 0; avoid warnings or NaN by discarding small S
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-14
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    def apply_operator(self,op:np.ndarray,*wires):
        """Applies the operator `op` at the given sites."""
        # sanity check
        assert np.allclose(op @ op.conj().T,np.eye(op.shape[0])), "op must be unitary!"
        assert op.shape[0] == op.shape[1], "op must be square!"

        if op.shape[0] == 2:
            # we got ourselves a single-qubit gate
            self.Bs[wires[0]] = np.einsum("ijk,jr->irk",self.Bs[wires[0]],op)
            return

        i1 = wires[0]
        i2 = wires[1]
        # sanity check
        assert i1 in range(self.L) and i2 in range(self.L), "i1 or i2 out of range!"
        assert i1 != i2, "i1 and i2 must be different!"

        if i2 < i1:
            # since the indices are flipped, we also need to flip the gate
            SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            op = SWAP @ op @ SWAP
            i1,i2 = i2,i1

        # right-most bond dimension (needed for re-shaping later)
        chiR = self.Bs[i2].shape[2]

        # re-shaping op
        op = np.reshape(op,newshape=(2,2,2,2))

        # we need to find the wavefunction that spans sites i1 to i2, so we multiply together the MPS in the range [i1,i2]
        psi = self.Bs[i1]
        iSite = i1 + 1
        while iSite <= i2:
            # contracting with the next bond singular values
            psi = np.einsum("...jk,k->...jk",psi,self.Ss[iSite])
            # contracting with the next site tensor
            psi = np.einsum("...i,ijk->...jk",psi,self.Bs[iSite])

            iSite += 1

        # applying the operator
        n_legs = len(psi.shape)
        psi = np.tensordot(op,psi,axes=((0,1),(1,-2)))
        new_axes = (2,0) + tuple(range(3,n_legs-1)) + (1,n_legs-1)
        # re-shaping psi (tensordor appends the non-contracted axes of the tensors)
        psi = np.transpose(psi,new_axes)

        new_Bs = []
        new_Ss = []
        # SVDs to re-establish the physical indices
        for i,iSite in enumerate(np.arange(i1,i2+1)):
            psi = np.reshape(psi,(-1,2**(i2 - i1 - i) * chiR))
            U,S,Vh = svd(psi,full_matrices=False)

            # omitting zero SVD values
            mask = S > 0
            U = U[:,mask]
            S = S[mask]
            Vh = Vh[mask,:]

            # re-normalizing the singular values
            norm = np.linalg.norm(S)

            new_Bs += [np.reshape(U,(-1,2,len(S))) * norm,]
            new_Ss += [S / norm,]

            psi = Vh
        # absorbing the last S and Vh into the last B
        new_Bs[-1] = np.einsum("ijk,k,kr->ijr",new_Bs[-1],new_Ss[-1],Vh)
        new_Ss.pop()

        # inserting the new bond singular values and site tensors back into the MPS
        self.Bs[i1:i2+1] = new_Bs
        self.Ss[i1+1:i2+1] = new_Ss

    def uncompress(self) -> np.ndarray:
        """Returns the state represented by the MPS in the computational basis."""
        psi = np.einsum("i,ijk->ijk",self.Ss[0],self.Bs[0])

        for i in range(1,self.L):
            psi = np.einsum("...k,k,kir->...ir",psi,self.Ss[i],self.Bs[i])

        return np.reshape(psi,(1,-1,1))[0,:,0]

def init_spinup_MPS(L):
    """Return a product state with all spins up as an MPS"""
    B = np.zeros([1, 2, 1], np.float64)
    B[0, 0, 0] = 1.
    S = np.ones([1], np.float64)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS(Bs, Ss)

def init_spinright_MPS(L):
    """Return a product state with all spins right as an MPS"""
    B = np.zeros([1, 2, 1], np.float64)
    B[0, :, 0] = np.ones(shape=(2,)) / np.sqrt(2)
    S = np.ones([1], np.float64)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS(Bs, Ss)

def split_truncate_theta(theta:np.ndarray,eps:float=0) -> list[np.ndarray,np.ndarray,np.ndarray]:
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : np.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : np.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : np.Array[ndim=1]
        Singular/Schmidt values.
    B : np.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    chivL, dL, dR, chivR = theta.shape
    theta = np.reshape(theta, [chivL * dL, dR * chivR])
    X, Y, Z = svd(theta, full_matrices=False)
    # truncate
    chivC = np.sum(Y > eps)
    assert chivC >= 1
    piv = np.argsort(Y)[::-1][:chivC]  # keep the largest `chivC` singular values
    X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
    # renormalize
    S = Y / np.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))
    # split legs of X and Z
    A = np.reshape(X, [chivL, dL, chivC])
    B = np.reshape(Z, [chivC, dR, chivR])
    return A, S, B
