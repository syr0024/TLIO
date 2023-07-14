"This lie_algebra file is from https://github.com/utiasSTARS/so3_learning."

import torch
import numpy as np

def allclose(mat1, mat2, tol=1e-10):
    """Check if all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return isclose(mat1, mat2, tol).all()


def isclose(mat1, mat2, tol=1e-10):
    """Check element-wise if two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return (mat1 - mat2).abs_().lt(tol)


def so3_wedge(phi):
    #Returns Nx3x3 tensor with each 1x3 row vector in phi wedge'd
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    Phi = phi.new_zeros(phi.shape[0], 3, 3)

    Phi[:, 0, 1] = -phi[:, 2]
    Phi[:, 1, 0] = phi[:, 2]
    Phi[:, 0, 2] = phi[:, 1]
    Phi[:, 2, 0] = -phi[:, 1]
    Phi[:, 1, 2] = -phi[:, 0]
    Phi[:, 2, 1] = phi[:, 0]
    return Phi

def so3_vee(Phi):
    #Returns Nx3 tensor with each 3x3 lie algebra element converted to a 1x3 coordinate vector
    if Phi.dim() < 3:
        Phi = Phi.unsqueeze(dim=0)
    phi = Phi.new_zeros(Phi.shape[0], 3)

    phi[:, 0] = Phi[:, 2, 1]
    phi[:, 1] = Phi[:, 0, 2]
    phi[:, 2] = Phi[:, 1, 0]
    return phi

def batch_trace(R):
    #Takes in Nx3x3, computes trace of each 3x3 matrix, outputs Nx1 vector with traces
    I = R.new(3,3).zero_()
    I[0,0] = I[1,1] = I[2,2] = 1.0
    return (R*I.expand_as(R)).sum(1, keepdim=True).sum(2, keepdim=True).view(R.size(0),-1)

def batch_outer_prod(vecs):
    #Input: NxD vectors
    #Output: NxDxD outer products
    N = vecs.size(0)
    D = vecs.size(1)
    return vecs.unsqueeze(2).expand(N,D,D)*vecs.unsqueeze(1).expand(N,D,D)

def so3_log(R):
    #input: R 64x3x3
    #output: log(R) 64x3

    batch_size = R.size(0)

    # The rotation axis (not unit-length) is given by
    axes = R.new(batch_size, 3).zero_()

    axes[:,0] = R[:, 2, 1] - R[:, 1, 2]
    axes[:,1] = R[:, 0, 2] - R[:, 2, 0]
    axes[:,2] = R[:, 1, 0] - R[:, 0, 1]


    # The cosine of the rotation angle is related to the trace of C
    #NOTE: clamp ensures that we don't get any nan's due to out of range numerical errors

    ## clamp 되는거 있는지 확인하는 코드
    # if torch.any(batch_trace(R)>3) or torch.any(batch_trace(R)<-1):
    #     print('[so3_log]')
    #     print('larger than 3: ', torch.count_nonzero(batch_trace(R)>3))
    #     print('smaller than -1: ', torch.count_nonzero(batch_trace(R)<-1))

    angles = torch.acos((0.5 * batch_trace(R) - 0.5).clamp(-1., 1.))
    sin_angles = torch.sin(angles)
    
    

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = isclose(angles, 0.).squeeze()
    small_angles_num = small_angles_mask.sum().item()
    pi_angles_mask = (np.pi - angles).lt(1e-7).view(-1)


    #This tensor is used to extract the 3x3 R's that correspond to small angles
    small_angles_indices = small_angles_mask.nonzero().squeeze()


    #print('small angles: {}/{}.'.format(small_angles_num, batch_size))

    if small_angles_num == 0:
        #Regular log
        ax_sin = axes / sin_angles.expand_as(axes)
        logs = 0.5 * angles.expand_as(ax_sin) * ax_sin

    elif small_angles_num == batch_size:
        #Small angle Log
        I = R.new(3, 3).zero_()
        I[0,0] = I[1,1] = I[2,2] = 1.0
        I = I.expand(batch_size, 3,3) #I is now batch_sizex3x3
        logs = so3_vee(R - I)
    else:
        #Some combination of both
        I = R.new(3, 3).zero_()
        I[0,0] = I[1,1] = I[2,2] = 1.0
        I = I.expand(small_angles_num, 3,3) #I is now small_angles_numx3x3

        ax_sin = (axes / sin_angles.expand_as(axes))
        logs = 0.5 * angles.expand_as(ax_sin) * ax_sin


        small_logs = so3_vee(R[small_angles_indices] - I)
        logs[small_angles_indices] = small_logs

    ## angle이 pi에 가까워지는 값이 있을 때 log 값이 무한대로 발산하지 않도록 다른 수식 사용
    if pi_angles_mask.any():
        # print('theta close to pi angle index: ', torch.nonzero(pi_angles_mask))
        # print('R: ', R[pi_angles_mask, :, :])
        # print('angles: ', angles[pi_angles_mask, :])
        # print('sin_angles: ', sin_angles[pi_angles_mask, :])
        print('Warning - theta close to pi - may get erroneous SO3() logs!')
        pi_angles_indices = pi_angles_mask.nonzero().squeeze(1)
        I = R.new(3, 3).zero_()
        I[0,0] = I[1,1] = I[2,2] = 1.0
        for i in range(pi_angles_indices.size(0)):
            pi_angles_R = R[pi_angles_indices[i],: ,:].squeeze()
            S = torch.sign(torch.tensor((pi_angles_R[0,1], pi_angles_R[0,2], pi_angles_R[1,2])))
            sign = torch.tensor((1, S[0], S[1]))

            logs[pi_angles_indices[i],:] = torch.tensor(((sign[0]*(0.5*(1+pi_angles_R[0,0]))**0.5)*np.pi,
                                                         (sign[1]*(0.5*(1+pi_angles_R[1,1]))**0.5)*np.pi,
                                                         (sign[2]*(0.5*(1+pi_angles_R[2,2]))**0.5)*np.pi))
            print(pi_angles_R)
            print(logs[pi_angles_indices[i]-1,:])
            print("pi_angles_log: ", logs[pi_angles_indices[i],:])
            print(logs[pi_angles_indices[i]+1,:])

    ## logarithm 해준 값들 중 pi 보다 큰 값이 있는지 확인
    log_mask = logs>np.pi
    if torch.any(torch.any(log_mask, dim=1)):
        print('theta close to pi angle index: ', torch.nonzero(torch.any(log_mask, dim=1)))
        print('R: ', R[torch.any(log_mask, dim=1), :, :])
        print('angles: ', angles[torch.any(log_mask, dim=1), :])
        print('sin_angles: ', sin_angles[torch.any(log_mask, dim=1), :])
        print('logs: ', logs[torch.any(log_mask, dim=1), :])
        #print('Warning - theta close to pi - may get erroneous SO3() logs!')

    # if pi_angles_mask.any():
    #     I = R.new(3, 3).zero_()
    #     I[0,0] = I[1,1] = I[2,2] = 1.0
    #     I = I.expand(pi_angles_mask.sum(), 3,3)
    #     logs[pi_angles_mask.nonzero().squeeze()] = PI*torch.ones(pi_angles_mask.sum(),3)*(0.57735026919) #diag(0.5*(R[pi_angles_mask] + I))

    return logs



def so3_exp(phi):
    #input: phi 64x3
    #output: perturbation 64x3x3

    batch_size = phi.size(0)
    
    #Take the norms of each row
    angles = vec_norms(phi)

    I = phi.new(3, 3).zero_()
    I[0,0] = I[1,1] = I[2,2] = 1.0
    I = I.expand(batch_size, 3,3) #I is now num_samplesx3x3

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = isclose(angles, 0.).squeeze()
    small_angles_num = small_angles_mask.sum().item()
    small_angles_indices = small_angles_mask.nonzero().squeeze()

    if small_angles_num == batch_size:
        #Taylor expansion
        I = I.expand(batch_size, 3,3) 
        phi_w = so3_wedge(phi)
        return I + phi_w



    axes = phi / angles.expand(batch_size, 3)
    s = torch.sin(angles)
    c = torch.cos(angles)

    #This piece of magic multiplies each 3x3 matrix by each element in c

    c = c.view(-1,1,1).expand_as(I)
    s = s.view(-1,1,1).expand_as(I)

    outer_prod_axes = batch_outer_prod(axes)
    R = c * I + (1 - c) * outer_prod_axes + s * so3_wedge(axes)

    if 0 < small_angles_num < batch_size:
        I_small = I[:small_angles_num]
        phi_w = so3_wedge(phi[small_angles_indices])
        small_exp = I_small + phi_w
        R[small_angles_indices] = small_exp
    

    return R
    

def vec_norms(input):
    #Takes Nx3 tensor of row vectors an outputs Nx1 tensor with 2-norms of each row
    norms = input.pow(2).sum(dim=1, keepdim=True).sqrt()
    return norms

def vec_square_norms(input):
    #Takes Nx3 tensor of row vectors an outputs Nx1 tensor with squared 2-norms of each row
    sq_norms = input.pow(2).sum(dim=1, keepdim=True)
    return sq_norms


def so3_inv_left_jacobian(phi):
    """Inverse left SO(3) Jacobian (see Barfoot).
    """
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    angles = vec_norms(phi)
    I = phi.new(3, 3).zero_()
    I[0,0] = I[1,1] = I[2,2] = 1.0
    batch_size = phi.size(0)

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = isclose(angles, 0.).squeeze()
    small_angles_num = small_angles_mask.sum().item()
    small_angles_indices = small_angles_mask.nonzero().squeeze()

    if small_angles_num == batch_size:
        I = I.expand(batch_size, 3,3)
        return I - 0.5*so3_wedge(phi)


    axes = phi / angles.expand(batch_size, 3)
    half_angles = 0.5 * angles
    cot_half_angles = 1. / torch.tan(half_angles)
    I_full = I.expand(batch_size, 3,3) #I is now num_samplesx3x3
  
    #Compute outer products of the Nx3 axes vectors, and put them into a Nx3x3 tensor
    outer_prod_axes = batch_outer_prod(axes)

    #This piece of magic changes the vector so that it can multiply each 3x3 matrix of a Nx3x3 tensor
    h_a_cot_a = (half_angles * cot_half_angles).view(-1,1,1).expand_as(I_full)
    h_a = half_angles.view(-1,1,1).expand_as(I_full)
    invJ = h_a_cot_a * I_full + (1 - h_a_cot_a) * outer_prod_axes - (h_a * so3_wedge(axes))

    if 0 < small_angles_num < batch_size:
        I_small = I.expand(small_angles_num, 3,3)
        small_invJ = I_small - 0.5*so3_wedge(phi[small_angles_indices])
        invJ[small_angles_indices] = small_invJ

    return invJ

def so3_left_jacobian(phi):
    """Inverse left SO(3) Jacobian (see Barfoot).
    """
    if phi.dim() < 2:
        phi = phi.unsqueeze(dim=0)

    angles = vec_norms(phi)
    I = phi.new(3, 3).zero_()
    I[0,0] = I[1,1] = I[2,2] = 1.0
    batch_size = phi.size(0)

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = isclose(angles, 0.).squeeze()
    small_angles_num = small_angles_mask.sum().item()
    small_angles_indices = small_angles_mask.nonzero().squeeze()

    if small_angles_num == batch_size:
        I = I.expand(batch_size, 3,3)
        return I + 0.5*so3_wedge(phi)
  
    axes = phi / angles.expand(batch_size, 3)
    #Compute outer products of the Nx3 axes vectors, and put them into a Nx3x3 tensor
    outer_prod_axes = batch_outer_prod(axes)

    sin_ph = torch.sin(angles)
    cos_ph = torch.cos(angles)

    I_full = I.expand(batch_size, 3,3)
    t1 = sin_ph/angles
    t2 = 1 - t1
    t3 = (1 - cos_ph)/angles

    t1 = t1.view(-1,1,1).expand_as(I_full)
    t2 = t2.view(-1,1,1).expand_as(I_full)
    t3 = t3.view(-1,1,1).expand_as(I_full)
    
    J = t1 * I_full + t2 * outer_prod_axes + t3 * so3_wedge(axes)

    if 0 < small_angles_num < batch_size:
        I_small = I.expand(small_angles_num, 3,3)
        small_J = I_small + 0.5*so3_wedge(phi[small_angles_indices])
        J[small_angles_indices] = small_J

    return J

    

def so3_to_rpy(rot):
    """Convert a rotation matrix to RPY Euler angles."""
    #Nx3x3 -> 3xN

    PI = 3.14159265358979323846

    pitch = torch.atan2(-rot[:, 2, 0],
                        torch.sqrt(rot[:, 0, 0]**2 + rot[:, 1, 0]**2))

    sec_pitch = 1. / torch.cos(pitch)
    yaw = torch.atan2(rot[:, 1, 0] * sec_pitch,
                        rot[:, 0, 0] * sec_pitch)
    roll = torch.atan2(rot[:, 2, 1] * sec_pitch,
                        rot[:, 2, 2] * sec_pitch)

    pospi2_mask = torch.abs(pitch - PI/2.0).lt(EPS).view(-1)
    pospi2_angles_num = pospi2_mask.sum()
    pospi2_indices = pospi2_mask.nonzero().squeeze()
    
    negpi2_mask = torch.abs(pitch + PI/2.0).lt(EPS).view(-1)
    negpi2_angles_num = negpi2_mask.sum()
    negpi2_indices = negpi2_mask.nonzero().squeeze()

    if pospi2_angles_num > 0:
        yaw[pospi2_indices] = 0.
        roll[pospi2_indices] = torch.atan2(rot[pospi2_indices, 0, 1], rot[pospi2_indices, 1, 1])

    if negpi2_angles_num > 0:
        yaw[pospi2_indices] = 0.
        roll[pospi2_indices] = -torch.atan2(rot[pospi2_indices, 0, 1], rot[pospi2_indices, 1, 1])
        

    return torch.cat((roll.view(-1,1), pitch.view(-1,1), yaw.view(-1,1)), 1)

def rpy_to_so3(rpy):
    """Convert RPY Euler angles to a rotation matrix."""
    #3xN -> Nx3x3
    roll = rpy[:,0].view(-1,1,1)
    pitch = rpy[:,1].view(-1,1,1)
    yaw = rpy[:,2].view(-1,1,1)
    
    c_r = torch.cos(roll)
    s_r = torch.sin(roll)

    c_p = torch.cos(pitch)
    s_p = torch.sin(pitch)

    c_y = torch.cos(yaw)
    s_y = torch.sin(yaw)

    rotz = rpy.new(rpy.size(0), 3, 3).zero_()
    rotz[:,2,2] = 1.0
    rotz[:,0,0] = rotz[:,1,1] = c_y
    rotz[:,0,1] = -s_y
    rotz[:,1,0] = s_y

    roty = rpy.new(rpy.size(0), 3, 3).zero_()
    roty[:,1,1] = 1.0
    roty[:,0,0] = roty[:,2,2] = c_p
    roty[:,0,2] = s_p
    roty[:,2,0] = -s_p
    
    rotx = rpy.new(rpy.size(0), 3, 3).zero_()
    rotx[:,0,0] = 1.0
    rotx[:,1,1] = rotz[:,2,2] = c_r
    rotx[:,1,2] = -s_r
    rotx[:,2,1] = s_r
    
    return rotz.bmm(roty.bmm(rotx))

#================================SE(3)====================================#
def se3_wedge(xi):
    #Returns Nx4x4 tensor with each 1x6 row vector in xi SE(3) wedge'd
    if xi.dim() < 2:
        xi = xi.unsqueeze(dim=0)
        
    n = xi.shape[0]
    Xi = xi.new_zeros(n, 4, 4)

    rho = xi[:, :3]
    phi = xi[:, 3:]
    Phi = so3_wedge(phi)

    Xi[:, :3, :3] = Phi
    Xi[:, :3, 3] = rho
    
    return Xi

def se3_curly_wedge(xi):
    #Returns Nx4x4 tensor with each 1x6 row vector in xi SE(3) curly wedge'd
    Xi = xi.new(xi.size(0), 6, 6).zero_()
    rho = xi[:, 0:3]
    phi = xi[:, 3:6]
    Phi = so3_wedge(phi)
    Rho = so3_wedge(rho)

    Xi[:, 0:3, 0:3] = Phi
    Xi[:, 0:3, 3:6] = Rho
    Xi[:, 3:6, 3:6] = Phi
    
    return Xi 

def se3_log(T):

    """Logarithmic map for SE(3)

    Computes a SE(3) tangent vector from a transformation
    matrix.

    This is the inverse operation to exp

    #input: T Nx4x4
    #output: log(T) Nx6

    """
    if T.dim() < 3:
        T = T.unsqueeze(dim=0)

    R = T[:,0:3,0:3]
    t = T[:,0:3,3:4]
    sample_size = t.size(0)
    phi = so3_log(R)
    invl_js = so3_inv_left_jacobian(phi)
    rho = (invl_js.bmm(t)).view(sample_size, 3)
    return torch.cat((rho, phi), 1).squeeze()

def se3_exp(xi):
    #input: xi Nx6
    #output: T Nx4x4
    #New efficient way without having to compute Q!

    if xi.dim() < 2:
        xi = xi.unsqueeze(dim=0)

    batch_size = xi.shape[0]
    phi = vec_norms(xi[:, 3:6])

    I = xi.new(4, 4).zero_()
    I[0,0] = I[1,1] = I[2,2] = I[3,3] = 1.0

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = isclose(phi, 0.).squeeze()
    small_angles_num = small_angles_mask.sum()
    small_angles_indices = small_angles_mask.nonzero().squeeze()


    if small_angles_num == batch_size:
        #Taylor expansion
        I = I.expand(batch_size, 4,4) 
        xi_w = se3_wedge(xi)
        return I + xi_w

    xi_w = se3_wedge(xi)
    xi_w2 = xi_w.bmm(xi_w)
    xi_w3 = xi_w2.bmm(xi_w)
    
    phi2 = phi.pow(2)
    phi3 = phi.pow(3)

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    I_full = I.expand(batch_size, 4,4) 
    t2 = (1 - cos_phi)/phi2
    t3 = (phi - sin_phi)/phi3
    t2 = t2.view(-1,1,1).expand_as(I_full)
    t3 = t3.view(-1,1,1).expand_as(I_full)

    T = I_full + xi_w + t2*xi_w2 + t3*xi_w3
    
    if 0 < small_angles_num < batch_size:
        I_small = I.expand(small_angles_num, 4,4)
        xi_w = se3_wedge(xi[small_angles_indices])
        small_exp = I_small + xi_w
        T[small_angles_indices] = small_exp
    
    return T


def se3_inv(T):
    #Batch invert Nx4x4 SE(3) matrices
    Rt = T[:, 0:3, 0:3].transpose(1,2)
    t = T[:, 0:3, 3:4]

    T_inv = T.clone()
    #T_inv = T.new(T.size()).zero_()
    #T_inv[:,3,3] = 1.0

    T_inv[:, 0:3, 0:3] = Rt
    T_inv[:, 0:3, 3:4] = -Rt.bmm(t) 
    
    return T_inv

def se3_Q(rho, phi):
    #SE(3) Q function
    #Used in the SE(3) jacobians
    #See b

    ph = vec_norms(phi)

    ph2 = ph*ph
    ph3 = ph2*ph
    ph4 = ph3*ph
    ph5 = ph4*ph

    rx = so3_wedge(rho)
    px = so3_wedge(phi)

    #Turn Nx1 into a Nx3x3 (with each 3x3 having 9 identical numbers)
    cph = torch.cos(ph).view(-1,1,1).expand_as(rx)
    sph = torch.sin(ph).view(-1,1,1).expand_as(rx)
    ph = ph.view(-1,1,1).expand_as(rx)
    ph2 = ph2.view(-1,1,1).expand_as(rx)
    ph3 = ph3.view(-1,1,1).expand_as(rx)
    ph4 = ph4.view(-1,1,1).expand_as(rx)
    ph5 = ph5.view(-1,1,1).expand_as(rx)
    
    m1 = 0.5
    m2 = (ph - sph)/ph3
    m3 = (ph2 + 2. * cph - 2.)/(2.*ph4)
    m4 = (2.*ph - 3.*sph + ph*cph)/(2.*ph5)

    t1 = m1 * rx
    t2 = m2 * (px.bmm(rx) + rx.bmm(px) + px.bmm(rx).bmm(px))
    t3 = m3 * (px.bmm(px).bmm(rx) + rx.bmm(px).bmm(px) - 3. * px.bmm(rx).bmm(px))
    t4 = m4 * (px.bmm(rx).bmm(px).bmm(px) + px.bmm(px).bmm(rx).bmm(px))

    Q = t1 + t2 + t3 + t4

    return Q

def se3_left_jacobian(xi):
    """Computes SE(3) left jacobian of N xi vectors (arranged into NxD tensor)"""
    if xi.dim() < 2:
        xi = xi.unsqueeze(dim=0)

    rho = xi[:, 0:3]
    phi = xi[:, 3:6]

    batch_size = xi.size(0)
    angles = vec_norms(xi[:, 3:6])

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = isclose(angles, 0.).squeeze()
    small_angles_num = small_angles_mask.sum()
    small_angles_indices = small_angles_mask.nonzero().squeeze()

    if small_angles_num == batch_size:
        #Taylor expansion
        I = xi.new(6, 6).zero_()
        I[0,0] = I[1,1] = I[2,2] = I[3,3] = I[4,4] = I[5,5] =  1.0
        I = I.expand(batch_size, 6,6) 
        return I + 0.5*se3_curly_wedge(xi)


    J = so3_left_jacobian(phi)
    Q = se3_Q(rho, phi)
    zero_mat = xi.new(3, 3).zero_().expand(batch_size, 3, 3)
  
    upper_rows = torch.cat((J, Q), 2)
    lower_rows = torch.cat((zero_mat, J), 2)
    
    J = torch.cat((upper_rows, lower_rows), 1)

    if 0 < small_angles_num < batch_size:
        I = xi.new(6, 6).zero_()
        I[0,0] = I[1,1] = I[2,2] = I[3,3] = I[4,4] = I[5,5] =  1.0
        I = I.expand(batch_size, 6,6) 
        small_J =  I + 0.5*se3_curly_wedge(xi[small_angles_indices])
        J[small_angles_indices] = small_J

    return J
    


def se3_inv_left_jacobian(xi):
    """Computes SE(3) inverse left jacobian of N xi vectors (arranged into NxD tensor)"""
    if xi.dim() < 2:
        xi = xi.unsqueeze(dim=0)

    rho = xi[:, 0:3]
    phi = xi[:, 3:6]

    batch_size = xi.size(0)
    angles = vec_norms(xi[:, 3:6])

    # If angle is close to zero, use first-order Taylor expansion
    small_angles_mask = isclose(angles, 0.).squeeze()
    small_angles_num = small_angles_mask.sum()
    small_angles_indices = small_angles_mask.nonzero().squeeze()

    if small_angles_num == batch_size:
        #Taylor expansion
        I = xi.new(6, 6).zero_()
        I[0,0] = I[1,1] = I[2,2] = I[3,3] = I[4,4] = I[5,5] =  1.0
        I = I.expand(batch_size, 6,6)  
        return I - 0.5*se3_curly_wedge(xi)

    invl_j = so3_inv_left_jacobian(phi)
    Q = se3_Q(rho, phi)
    zero_mat = xi.new(3, 3).zero_().expand(batch_size, 3, 3)
  
    upper_rows = torch.cat((invl_j, -invl_j.bmm(Q).bmm(invl_j)), 2)
    lower_rows = torch.cat((zero_mat, invl_j), 2)
    
    inv_J = torch.cat((upper_rows, lower_rows), 1)

    if 0 < small_angles_num < batch_size:
        I = xi.new(6, 6).zero_()
        I[0,0] = I[1,1] = I[2,2] = I[3,3] = I[4,4] = I[5,5] =  1.0
        I = I.expand(batch_size, 6,6) 
        small_inv_J =  I - 0.5*se3_curly_wedge(xi[small_angles_indices])
        inv_J[small_angles_indices] = small_inv_J

    return inv_J


def se3_adjoint(T):
    if T.dim() < 2:
        T = T.unsqueeze(dim=0)
    
    C = T[:, :3, :3]
    Jrho_wedge = so3_wedge(T[:, :3, 3].view(-1, 3))

    adj_T = T.new_zeros((T.shape[0], 6,6))
    adj_T[:, :3, :3] = C
    adj_T[:, :3, 3:] = Jrho_wedge.bmm(C)
    adj_T[:, 3:, 3:] = C

    return adj_T
    


    
