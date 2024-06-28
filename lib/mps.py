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
        """Returns a copy of itself."""
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
            S = self.Ss[i].copy().real # singular values are real anyways
            S = S[S > 1e-30]  # 0*log(0) should give 0; avoid warnings or NaN by discarding small S
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-14
            result.append(-np.sum(S2 * np.log2(S2)))
        return np.array(result)

    def apply_operator(self,op:np.ndarray,*wires,eps:float=1e-12,chi_max:int=None):
        """Applies the operator `op` at the given sites."""
        # sanity check
        assert np.allclose(op @ op.conj().T,np.eye(N=op.shape[0])), "op must be unitary!"
        assert op.shape[0] == op.shape[1], "op must be square!"

        if op.shape[0] == 2:
            # we got ourselves a single-qubit gate
            self.Bs[wires[0]] = np.einsum("rj,ijk->irk",op,self.Bs[wires[0]])
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
        psi = np.einsum("i,ijk->ijk",self.Ss[i1],self.Bs[i1])
        for iSite in range(i1+1,i2+1): psi = np.einsum("...i,ijk->...jk",psi,self.Bs[iSite])

        # applying the operator
        n_legs = len(psi.shape)
        psi = np.tensordot(op,psi,axes=((2,3),(1,-2)))
        new_axes = (2,0) + tuple(range(3,n_legs-1)) + (1,n_legs-1)
        # re-shaping psi (tensordot appends the non-contracted axes of the tensors)
        psi = np.transpose(psi,new_axes)

        new_Bs = []
        new_Ss = []
        # SVDs to re-establish the physical indices
        for i,iSite in enumerate(np.arange(i1,i2)):
            psi = np.reshape(psi,(-1,2**(i2 - i1 - i) * chiR))
            U,S,Vh = svd(psi,full_matrices=False)

            if chi_max != None:
                # truncating singular values
                U = U[:,:chi_max]
                S = S[:chi_max]
                Vh = Vh[:chi_max,:]

            # omitting close-to-zero SVD values
            mask = S > eps
            U = U[:,mask]
            S = S[mask]
            Vh = Vh[mask,:]

            new_Bs += [np.reshape(U,(-1,2,len(S))),]
            new_Ss += [S,]

            psi = Vh.copy()

        new_Bs += [np.reshape(Vh,(len(S),2,-1)),]

        # mutliplying the S left to B[i1] into new_Bs[0] to preserve the canonical form
        new_Bs[0] = np.einsum("i,ijk->ijk",1 / self.Ss[i1],new_Bs[0])

        # multiplying the next S into the previous B to preserve the canonical form
        for i,S in enumerate(new_Ss):
            new_Bs[i] = np.einsum("ijk,k->ijk",new_Bs[i],S)

        # inserting the new B's and S's back into the MPS
        self.Bs[i1:i2+1] = new_Bs
        self.Ss[i1+1:i2+1] = new_Ss

        return

    def uncompress(self) -> np.ndarray:
        """Returns the state represented by the MPS in the computational basis."""
        psi = np.array([1,])

        for i in range(0,self.L):
            psi = np.einsum("...k,kir->...ir",psi,self.Bs[i])

        return np.reshape(psi,(1,-1,1))[0,:,0]

    def check_normalization(self):
        print("Checking normalization.\n  Norm of complete MPS = {:.3e}".format(np.linalg.norm(self.uncompress())))
        for i1 in range(self.L):
            for i2 in range(i1+1,self.L):
                # finding the wavefunction that spans sites i1 to i2, so we multiply together the MPS in the range [i1,i2]
                psi = np.einsum("i,ijk->ijk",self.Ss[i1],self.Bs[i1])
                for iSite in range(i1+1,i2+1): psi = np.einsum("...i,ijk->...jk",psi,self.Bs[iSite])

                if not np.isclose(np.linalg.norm(psi),1):
                    print("  psi[{},{}] is not normalized! |psi[{},{}]| = {:.3e}".format(i1,i2,i1,i2,np.linalg.norm(psi)))

    def __repr__(self):
        info = f"MPS over {self.L} sites.\n"
        info += "    site tensor shapes: "
        for B in self.Bs: info += (str(B.shape) + " ")
        info += "\n    bond dimensions: "
        for S in self.Ss: info += (str(len(S)) + " ")
        info += "\n    norm: {:.3e}".format(np.linalg.norm(self.uncompress()))

        # diagnosing non-normalized singular values
        info += "\n    Any bond dimensions not normalized? "
        for i,S in enumerate(self.Ss):
            if abs(np.linalg.norm(S) - 1.) > 1.e-14:
                info += f" bond dimension {i} not normalized. "

        return info

def init_spinup_MPS(L):
    """Return a product state with all spins up as an MPS"""
    B = np.zeros(shape=[1, 2, 1])
    B[0, 0, 0] = 1.
    S = np.ones(shape=[1])
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS(Bs, Ss)

def init_spinright_MPS(L):
    """Return a product state with all spins right as an MPS"""
    B = np.zeros(shape=[1, 2, 1])
    B[0, :, 0] = np.ones(shape=(2,)) / np.sqrt(2)
    S = np.ones(shape=[1,])
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
