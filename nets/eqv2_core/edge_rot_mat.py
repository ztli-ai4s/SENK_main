import torch


def _legacy_edge_rot_mat(edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    #assert torch.min(edge_vec_0_distance) < 0.0001
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            "Error edge_vec_0_distance: {}".format(
                torch.min(edge_vec_0_distance)
            )
        )
        
    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
        -1, 1
    )
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
        -1, 1
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat


def _stable_edge_rot_mat(edge_distance_vec, eps=1.0e-8):
    """Deterministic and numerically safe local frame construction.

    This path is used when rotation-to-Wigner needs autograd connection to
    edge vectors (e.g., Hij). It avoids random auxiliary vectors and protects
    all normalizations with eps.
    """
    if edge_distance_vec.numel() == 0:
        return edge_distance_vec.new_zeros((0, 3, 3))

    vec = edge_distance_vec
    dist = torch.sqrt(torch.sum(vec * vec, dim=1, keepdim=True)).clamp_min(eps)
    norm_x = vec / dist

    # For pathological near-zero edges, use a fixed fallback direction.
    near_zero = (torch.sum(vec * vec, dim=1, keepdim=True) <= (eps * eps))
    fallback_x = vec.new_tensor([1.0, 0.0, 0.0]).view(1, 3).expand_as(norm_x)
    norm_x = torch.where(near_zero, fallback_x, norm_x)

    ref_a = vec.new_tensor([0.0, 0.0, 1.0]).view(1, 3).expand_as(norm_x)
    ref_b = vec.new_tensor([0.0, 1.0, 0.0]).view(1, 3).expand_as(norm_x)
    cross_a = torch.cross(norm_x, ref_a, dim=1)
    cross_b = torch.cross(norm_x, ref_b, dim=1)
    use_b = (torch.sum(cross_b * cross_b, dim=1, keepdim=True)
             > torch.sum(cross_a * cross_a, dim=1, keepdim=True))
    norm_z = torch.where(use_b, cross_b, cross_a)
    norm_z = norm_z / torch.sqrt(torch.sum(norm_z * norm_z, dim=1, keepdim=True)).clamp_min(eps)

    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.sqrt(torch.sum(norm_y * norm_y, dim=1, keepdim=True)).clamp_min(eps)

    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)
    return edge_rot_mat


def init_edge_rot_mat(edge_distance_vec, detach=True, stable=False, eps=1.0e-8):
    if stable:
        edge_rot_mat = _stable_edge_rot_mat(edge_distance_vec, eps=eps)
    else:
        edge_rot_mat = _legacy_edge_rot_mat(edge_distance_vec)
    if detach:
        return edge_rot_mat.detach()
    return edge_rot_mat