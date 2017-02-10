# -*- coding: utf-8 -*-
"""
A method to solve the advection equation on the inverse diffeomorphism
This is the corner transport upwind method on a three dimensional regular
grid. Limited high resolution corrections are added with three options:
min-mod, super-bee, and MC

Author: Greg M. Fleishman
"""

import numpy as np


# q is a displacement vector field being advected by velocity v
# assume q and v are defined at the center of cells
# vox is the voxel dimenions, dt is the duration of the time step
# _t is a transformer object to help find velocity at cell walls
def solve_advection_ctu(q, v, vox, dt, _t):
    """Solve the advection equation on the inverse diffeomorphism"""

    # convenient slice objects used throughout
    so_left = slice(None, -1, None)
    so_right = slice(1, None, None)
    so_all = slice(None, None, None)

    # get image shape and dimension
    sh = q.shape[:-1]
    d = q.shape[-1]

    # convert displacement vector field to position vector field
    q_next = np.copy(q)

    # find signed normal velocity component at center of wave fronts
    sh_plus = tuple(np.array(sh) + 1)
    signed_vel = np.zeros((d, 2) + sh_plus)
    position = _t.position_array(sh, vox)
    disp = np.copy(position)
    pad_array = [(0, 1)]*d
    for i in range(d):
        disp[..., i] -= 0.5 * vox[i]
        vel = _t.apply_transform(v[..., i], vox, disp)
        vel = np.pad(vel, pad_array, mode='constant')
        disp[..., i] = position[..., i]
        less = vel < 0
        signed_vel[i, 0][less] = vel[less]
        signed_vel[i, 1][~less] = vel[~less]
    del disp

    # compute waves
    waves = np.empty((d,) + sh_plus + (d,))
    for i in range(d):
        so_l = [so_all]*(d+1)
        so_l[i] = so_left
        so_r = [so_all]*(d+1)
        so_r[i] = so_right
        pad_array = [(0, 1)]*d + [(0, 0)]
        pad_array[i] = (1, 1)
        waves[i] = np.pad(q_next[so_r] - q_next[so_l],
                          pad_array, mode='constant')

    # update solution with donor-cell fluxes
    for i in range(d):
        so_v = [i, 1] + [so_left]*d
        so_w = [i] + [so_left]*d + [so_all]
        q_next -= dt/vox[i] * signed_vel[so_v][..., np.newaxis] * waves[so_w]

        so_v = [i, 0] + [so_left]*d
        so_v[i+2] = so_right
        so_w[i+1] = so_right
        q_next -= dt/vox[i] * signed_vel[so_v][..., np.newaxis] * waves[so_w]

    # update solution with corner transport upwind correction fluxes
    # 2D solution
    if d == 2:
        for i in range(d):
            # helpful slice objects for later, flux term constant
            oi = (i + 1) % 2
            so_bottom = [so_all]*(d+1)
            so_bottom[oi] = so_left
            so_top = [so_all]*(d+1)
            so_top[oi] = so_right
            C = 0.5*dt**2/np.prod(vox)

            # right moving then left moving waves
            for j in range(2):
                so_v = [i, 1] + [so_left]*d
                so_w = [i] + [so_left]*d + [so_all]
                if j == 1:
                    # switch to left moving waves
                    so_v = [i, 0] + [so_left]*d
                    so_v[i+2] = so_right
                    so_w[i+1] = so_right
                w = signed_vel[so_v][..., np.newaxis] * waves[so_w]

                # ... being pulled up
                so_v = [oi, 1] + [so_left]*d
                so_v[oi+2] = so_right
                flux = C * signed_vel[so_v][..., np.newaxis] * w
                q_next += flux
                q_next[so_top] -= flux[so_bottom]

                # ... being pulled down
                so_v = [oi, 0] + [so_left]*d
                flux = C * signed_vel[so_v][..., np.newaxis] * w
                q_next -= flux
                q_next[so_bottom] += flux[so_top]

    # 3D solution
    if d == 3:
        for i in range(d):
            # helpful slice objects for later, and flux term constants
            oi1 = (i + 1) % 3
            oi2 = (i + 2) % 3
            so_bottom_oi1 = [so_all]*(d+1)
            so_bottom_oi1[oi1] = so_left
            so_top_oi1 = [so_all]*(d+1)
            so_top_oi1[oi1] = so_right
            so_bottom_oi2 = [so_all]*(d+1)
            so_bottom_oi2[oi2] = so_left
            so_top_oi2 = [so_all]*(d+1)
            so_top_oi2[oi2] = so_right

            so_bottom_oi1_bottom_oi2 = [so_all]*(d+1)
            so_bottom_oi1_bottom_oi2[oi1] = so_left
            so_bottom_oi1_bottom_oi2[oi2] = so_left
            so_bottom_oi1_top_oi2 = [so_all]*(d+1)
            so_bottom_oi1_top_oi2[oi1] = so_left
            so_bottom_oi1_top_oi2[oi2] = so_right

            so_top_oi1_bottom_oi2 = [so_all]*(d+1)
            so_top_oi1_bottom_oi2[oi1] = so_right
            so_top_oi1_bottom_oi2[oi2] = so_left
            so_top_oi1_top_oi2 = [so_all]*(d+1)
            so_top_oi1_top_oi2[oi1] = so_right
            so_top_oi1_top_oi2[oi2] = so_right

            C1 = 0.5*dt**2/np.prod(vox)
            C2 = (1./6.)*dt**3/np.prod(vox)

            # right moving then left moving waves
            for j in range(2):
                so_v = [i, 1] + [so_left]*d
                so_w = [i] + [so_left]*d + [so_all]
                if j == 1:
                    # switch to left moving waves
                    so_v = [i, 0] + [so_left]*d
                    so_v[i+2] = so_right
                    so_w[i+1] = so_right
                w = signed_vel[so_v][..., np.newaxis] * waves[so_w]

                # ... being pulled up...
                so_v = [oi1, 1] + [so_left]*d
                so_v[oi1+2] = so_right
                w2 = signed_vel[so_v][..., np.newaxis] * w
                flux = C1 * vox[oi2] * w2
                q_next += flux
                q_next[so_top_oi1] -= flux[so_bottom_oi1]

                #         ... and forward...
                so_v = [oi2, 1] + [so_left]*d
                so_v[oi1+2] = so_right
                so_v[oi2+2] = so_right
                flux = C2 * signed_vel[so_v][..., np.newaxis] * w2
                q_next -= flux
                q_next[so_top_oi1] += 2. * flux[so_bottom_oi1]
                q_next[so_top_oi1_top_oi2] -= flux[so_bottom_oi1_bottom_oi2]

                #         ... and backward...
                so_v = [oi2, 0] + [so_left]*d
                so_v[oi1+2] = so_right
                flux = C2 * signed_vel[so_v][..., np.newaxis] * w2
                q_next += flux
                q_next[so_top_oi1] -= 2. * flux[so_bottom_oi1]
                q_next[so_top_oi1_bottom_oi2] += flux[so_bottom_oi1_top_oi2]

                # ... being pulled down...
                so_v = [oi1, 0] + [so_left]*d
                w2 = signed_vel[so_v][..., np.newaxis] * w
                flux = C1 * vox[oi2] * w2
                q_next -= flux
                q_next[so_bottom_oi1] += flux[so_top_oi1]

                # create view into signed_vel with bottom pad for oi1 axis
                so_v = [oi2] + [so_all]*(d+1)
                so_v2 = list(so_v)
                so_v[oi1+2] = slice(-1, None, None)
                so_v2[oi1+2] = slice(None, -1, None)
                shifted_sv = np.concatenate((signed_vel[so_v],
                                             signed_vel[so_v2]), axis=oi1+1)

                #         ... and forward...
                so_v = [1] + [so_left]*d
                so_v[oi2+1] = so_right
                flux = C2 * shifted_sv[so_v][..., np.newaxis] * w2
                q_next += flux
                q_next[so_bottom_oi1_top_oi2] -= flux[so_top_oi1_bottom_oi2]

                #         ... and backward...
                so_v = [0] + [so_left]*d
                flux = C2 * shifted_sv[so_v][..., np.newaxis] * w2
                q_next -= flux
                q_next[so_bottom_oi1_bottom_oi2] += flux[so_top_oi1_top_oi2]

                # ... being pulled forward...
                so_v = [oi2, 1] + [so_left]*d
                so_v[oi2+2] = so_right
                w2 = signed_vel[so_v][..., np.newaxis] * w
                flux = C1 * vox[oi1] * w2
                q_next += flux
                q_next[so_top_oi2] -= flux[so_bottom_oi2]

                #         ... and up...
                so_v = [oi1, 1] + [so_left]*d
                so_v[oi2+2] = so_right
                so_v[oi1+2] = so_right
                flux = C2 * signed_vel[so_v][..., np.newaxis] * w2
                q_next -= flux
                q_next[so_top_oi2] += 2. * flux[so_bottom_oi2]
                q_next[so_top_oi1_top_oi2] -= flux[so_bottom_oi1_bottom_oi2]

                #         ... and down...
                so_v = [oi1, 0] + [so_left]*d
                so_v[oi2+2] = so_right
                flux = C2 * signed_vel[so_v][..., np.newaxis] * w2
                q_next += flux
                q_next[so_top_oi2] -= 2. * flux[so_bottom_oi2]
                q_next[so_bottom_oi1_top_oi2] += flux[so_top_oi1_bottom_oi2]

                # ... being pulled backward...
                so_v = [oi2, 0] + [so_left]*d
                w2 = signed_vel[so_v][..., np.newaxis] * w
                flux = C1 * vox[oi1] * w2
                q_next -= flux
                q_next[so_bottom_oi2] += flux[so_top_oi2]

                # create view into signed_vel with bottom pad for oi1 axis
                so_v = [oi1] + [so_all]*(d+1)
                so_v2 = list(so_v)
                so_v[oi2+2] = slice(-1, None, None)
                so_v2[oi2+2] = slice(None, -1, None)
                shifted_sv = np.concatenate((signed_vel[so_v],
                                             signed_vel[so_v2]), axis=oi2+1)

                #         ... and up...
                so_v = [1] + [so_left]*d
                so_v[oi1+1] = so_right
                flux = C2 * shifted_sv[so_v][..., np.newaxis] * w2
                q_next += flux
                q_next[so_top_oi1_bottom_oi2] -= flux[so_bottom_oi1_top_oi2]

                #         ... and down...
                so_v = [0] + [so_left]*d
                flux = C2 * shifted_sv[so_v][..., np.newaxis] * w2
                q_next -= flux
                q_next[so_bottom_oi1_bottom_oi2] += flux[so_top_oi1_top_oi2]

    # Make limited high resolution corrections
    for i in range(d):
        pad_array = [(0, 0)]*(d+1)
        pad_array[i] = (1, 1)
        aug_waves = np.pad(waves[i], pad_array, mode='constant')
        so_n = [so_all]*(d+1)
        so_d = [so_all]*(d+1)
        so_n[i] = slice(None, -2, None)
        so_d[i] = slice(1, -1, None)
        aug_waves_l = aug_waves[so_n]/aug_waves[so_d]
        so_n[i] = slice(2, None, None)
        aug_waves_r = aug_waves[so_n]/aug_waves[so_d]

        less = signed_vel[i, 0] < 0
        more = signed_vel[i, 1] > 0
        theta = np.zeros_like(waves[i])
        theta[more] = aug_waves_l[more]
        theta[less] = aug_waves_r[less]
        theta[np.isinf(theta)] = 0
        theta[np.isnan(theta)] = 0

        # min-mod
#        theta[theta > 1] = 1
#        theta[theta < 0] = 0

        # super-bee
#        theta = np.maximum(np.minimum(1, 2*theta), np.minimum(2, theta))
#        theta = np.maximum(0, theta)

        # MC
        theta = np.minimum((1+theta)/2, np.minimum(2, 2*theta))
        theta[theta < 0] = 0

        vel_abs = np.abs(signed_vel[i, 0] + signed_vel[i, 1])
        C = 0.5*vel_abs*(1 - dt/vox[i]*vel_abs)
        flux = dt/vox[i] * theta * C[..., np.newaxis] * waves[i]
        so = [so_left]*d + [so_all]
        q_next += flux[so]
        so[i] = so_right
        q_next -= flux[so]

    # return solution
    return q_next
