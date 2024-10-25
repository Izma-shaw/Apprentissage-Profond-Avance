#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2014 Romain HERAULT <romain.herault@insa-rouen.fr>
#

import numpy as np

import particlefilter

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

filependulum = 'pendulum.npz'


def f(x, param):
    ret = np.zeros((2,))
    ret[0] = x[0] + x[1] * param['dt']
    ret[1] = x[1] - param['g'] * np.sin(x[0]) * param['dt']
    return ret


def h(x, param):
    ret = np.zeros((1,))
    ret[0] = np.sin(x[0])
    return ret


def getQ(param):
    Q = np.zeros((2, 2))
    Q[0, 0] = (param['q'] * (param['dt']**3)) / 3.
    Q[1, 0] = (param['q'] * (param['dt']**2)) / 2.
    Q[0, 1] = (param['q'] * (param['dt']**2)) / 2.
    Q[1, 1] = param['q'] * param['dt']

    return Q


def performsFiltering(filein):

    data = np.load(filein)

    time = data['time']
    states = data['states']
    observations = data['observations']
    anglemeasurements = data['anglemeasurements']
    param = dict(zip(['dt', 'q', 'r', 'g', 'm0', 'p0'],
                     [data['dt'], data['q'], data['r'], data['g'], data['m0'], data['p0']]))

    n = observations.shape[0]

    def transition_function(x): return f(x, param)

    def observation_function(x): return h(x, param)

    transition_covariance = getQ(param)
    observation_covariance = np.atleast_2d(np.array([param['r']]))

    start_mean = param['m0']
    start_covariance = param['p0']

    # PF
    nParticles = 10
    np.random.seed(3)
    pf = particlefilter.ParticleFilter(transition_function, transition_covariance,
                                       observation_function, observation_covariance,
                                       start_mean, start_covariance)

    (particles, weights) = pf.forward(nParticles, observations,resampling_threshold=3.5)

    max_estimate = pf.estimateStatesFromMax(particles, weights)
    mean_estimate = pf.estimateStatesFromMean(particles, weights)
    # END PF

    plt.figure()
    plt.plot(time, states[:, 0], 'k-', linewidth=2)
    plt.plot(time[1:], anglemeasurements, 'r+')
    plt.plot(time, max_estimate[:, 0], 'b-', linewidth=1)
    plt.plot(time, mean_estimate[:, 0], 'g-', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Angle x_1')
    plt.ylim([-3, 4])
    plt.legend(['True angle', 'Measurements', 'Max Estimate', 'Mean Estimate'])
    plt.savefig('pendulumPF.pdf')
    plt.close()

    plt.figure()
    plt.plot(time, states[:, 0], 'k-', linewidth=2)
    for i in range(len(time)):
        plt.plot([time[i]] * nParticles, particles[i, :, 0], 'g,')
    plt.xlabel('Time')
    plt.ylabel('Angle x_1')
    plt.ylim([-3, 4])
    plt.savefig('pendulumPFparticules.pdf')
    plt.close()


def main():
    performsFiltering(filependulum)


if __name__ == "__main__":
    main()
