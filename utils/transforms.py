from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.conversions import (
    QuaternionCoeffOrder,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)

# Instead of importing torchtyping, simply use torch.Tensor.
TensorType = torch.Tensor


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def qsvec2rotmat_batched(qvec: torch.Tensor, svec: torch.Tensor) -> torch.Tensor:
    # unscaled_rotmat: compute the rotation matrix from the quaternion qvec.
    unscaled_rotmat = quaternion_to_rotation_matrix(qvec, QuaternionCoeffOrder.WXYZ)

    # Multiply the rotation matrix by the scale vector (unsqueezed appropriately).
    rotmat = svec.unsqueeze(-2) * unscaled_rotmat
    return rotmat


def rotmat2wxyz(rotmat):
    return rotation_matrix_to_quaternion(rotmat, order=QuaternionCoeffOrder.WXYZ)


def qvec2rotmat_batched(qvec: torch.Tensor) -> torch.Tensor:
    return quaternion_to_rotation_matrix(qvec, QuaternionCoeffOrder.WXYZ)


def qsvec2covmat_batched(qvec: torch.Tensor, svec: torch.Tensor) -> torch.Tensor:
    rotmat = qsvec2rotmat_batched(qvec, svec)
    return torch.bmm(rotmat, rotmat.transpose(-1, -2))
