#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as npl


def multivariateNormalLawPDF(mu, S, x):
    """
    Compute the pdf at point(s) x of the multivariate normal law parametrized by (mu,S)
    :param mu: law center
    :type mu: (n,) or (m,n) numpy.ndarray
    :param S: law covariance
    :type S: (n,n) numpy.ndarray
    :param x: points to analyse
    :type x: (n,) or (m,n) numpy.ndarray
    :returns: pdf
    :rtype: (1,) or (m,) numpy.Array
    """

    assert (type(mu) is np.ndarray), 'mu must be a numpy.ndarray'
    assert (type(S) is np.ndarray), 'S must be a numpy.ndarray'
    assert (type(x) is np.ndarray), 'x must be a numpy.ndarray'

    assert (len(mu.shape) == 1) or (len(mu.shape) == 2), \
        'mu must be a (n,) or (m,n) numpy.ndarray. Given mu shape: {}'.format(
            mu.shape)

    if len(mu.shape) > 1:
        n = mu.shape[1]
    else:
        n = mu.shape[0]

    assert not((len(mu.shape) == 2) and (len(x.shape) == 2)), \
        'mu and x can not be 2D numpy.ndarrays together. Given mu shape: {}, x shape: {}'.format(
            mu.shape, x.shape)

    assert (len(S.shape) == 2) and (S.shape[0] == n) and (S.shape[1] == n), \
        'S must be a (n,n) numpy.ndarray. Given S shape: {}, desired n: {}'.format(
            S.shape, n)

    assert (((len(x.shape) == 2) and (x.shape[1] == n)) or ((len(x.shape) == 1) and (x.shape[0] == n))), \
        'x must be a (n,) or (m,n) numpy.ndarray. Given x shape: {}, desired n: {}'.format(
            x.shape, n)

    norm = np.sqrt(np.power(2 * np.pi, n) * npl.det(S))

    diff = x - mu
    invS = npl.inv(S)

    def expfunc(d): return np.exp(-0.5 * d.dot(invS).dot(d.T))
    if len(diff.shape) == 2:
        expo = np.apply_along_axis(expfunc, 1, diff)
    else:
        expo = expfunc(diff)

    return expo / norm


class ParticleFilter(object):
    def __init__(self,
                 transition_function, transition_covariance,
                 observation_function, observation_covariance,
                 start_mean, start_covariance):
        """
            Instantiate a particle filter

            :param transition_function: python function that applies one step in the dynamical model
            :type transition_function: (nStateDim,) numpy.ndarray -> (nStateDim,) numpy.ndarray
            :param transition_covariance: covariance of the additive noise in the dynamical model
            :type transition_covariance: (nStateDim,nStateDim) numpy.ndarray
            :param observation_function: python function that applies the observation model on a given state
            :type observation_function: (nStateDim,) numpy.ndarray -> (nObservationDim,) numpy.ndarray
            :param observation_covariance: covariance of the additive noise in the observation model
            :type observation_covariance: (nObservationDim,nObservationDim) numpy.ndarray
            :param start_state_mean: current state mean
            :type start_state_mean: (nStateDim,) numpy.ndarray
            :param start_state_covariance: current state covariance
            :type start_state_covariance: (nStateDim,nStateDim) numpy.ndarray
        """
        assert (type(transition_covariance)
                is np.ndarray), 'transition_covariance must be a numpy.ndarray'
        assert (type(observation_covariance)
                is np.ndarray), 'observation_covariance must be a numpy.ndarray'
        assert (type(start_mean) is np.ndarray), 'start_mean must be a numpy.ndarray'
        assert (type(start_covariance)
                is np.ndarray), 'start_covariance must be a numpy.ndarray'

        assert (len(start_mean.shape) == 1),\
            'start_mean must be a 1D numpy.ndarray. Given start_mean shape: {}'.format(
                start_mean.shape)

        self.stateDim = start_mean.shape[0]

        assert (len(start_covariance.shape) == 2),\
            'start_covariance must be a 2D numpy.ndarray. Given start_covariance shape: {}'.format(
                start_covariance.shape)
        assert (start_covariance.shape[0] == self.stateDim and start_covariance.shape[1] == self.stateDim),\
            'start_covariance must be a (nStateDim,nStateDim) numpy.ndarray. Given start_covariance shape: {}, desired state dimension: {}'.format(
                start_covariance.shape, self.stateDim)

        assert (len(transition_covariance.shape) == 2),\
            'transition_covariance must be a 2D numpy.ndarray. Given transition_covariance shape: {}'.format(
                transition_covariance.shape)
        assert (transition_covariance.shape[0] == self.stateDim and transition_covariance.shape[1] == self.stateDim),\
            'transition_covariance must be a (nStateDim,nStateDim) numpy.ndarray. Given transition_covariance shape: {}, desired state dimension: {}'.format(
                transition_covariance.shape, self.stateDim)

        assert (len(observation_covariance.shape) == 2),\
            'observation_covariance must be a 2D numpy.ndarray. Given observation_covariance shape: {}'.format(
                observation_covariance.shape)
        assert (observation_covariance.shape[0] == observation_covariance.shape[1]),\
            'observation_covariance must be a (nObservationDim,nObservationDim) numpy.ndarray. Given observation_covariance shape: {}'.format(
                observation_covariance.shape)

        self.obsDim = observation_covariance.shape[0]

        self.transition_function = transition_function
        self.transition_covariance = transition_covariance
        self.observation_function = observation_function
        self.observation_covariance = observation_covariance
        self.start_mean = start_mean
        self.start_covariance = start_covariance

    def _applyTransition(self, particles):
        """
            Applies transition step to particles

            :param particles: particles to apply the transition step to
            :type particles: (nParticles,nStateDim) numpy.ndarray
            :returns: new particles
            :rtype: (nParticles,nStateDim) numpy.ndarray

        """
        assert (type(particles) is np.ndarray), 'particles must be a numpy.ndarray'
        assert (len(particles.shape) == 2),\
            'particles must be a 2D numpy.ndarray. Given particles shape: {}'.format(
                particles.shape)
        assert (particles.shape[1] == self.stateDim),\
            'particles must be a (nParticles,nStateDim)). Given particles shape: {}, desired state dimension: {}'.format(
                particles.shape, self.stateDim)

        nParticles = particles.shape[0]

        # Compute the deterministic part of the transition model
        # deterministic= ?
        raise NotImplementedError
        # Compute the stochastic part of the transition model
        stochastic = np.random.multivariate_normal(
            np.zeros((self.stateDim,)), self.transition_covariance, nParticles)

        return deterministic + stochastic

    def _computeWeights(self, particles, weights, observation):
        """
            According to the new observation, computes new particles weights

            :param particles: particles
            :type particles: (nParticles,nStateDim) numpy.ndarray
            :param weights: old weights
            :type weights: (nParticles,) numpy.ndarray
            :param observation: current observation
            :type observation: (nObservationDim,) numpy.ndarray
            :returns: new weights
            :rtype: (nParticles,) numpy.ndarray

        """

        assert (type(particles) is np.ndarray), 'particles must be a numpy.ndarray'
        assert (type(weights) is np.ndarray), 'weights must be a numpy.ndarray'
        assert (type(observation)
                is np.ndarray), 'observation must be a numpy.ndarray'

        assert (len(particles.shape) == 2),\
            'particles must be a 2D numpy.ndarray. Given particles shape: {}'.format(
                particles.shape)
        assert (particles.shape[1] == self.stateDim),\
            'particles must be a (nParticles,nStateDim) numpy.ndarray. given particles shape: {}, desired state dimension: {}'.format(
                particles.shape, self.stateDim)
        nParticles = particles.shape[0]

        assert (len(weights.shape) == 1),\
            'weights must be a 1D numpy.ndarray. Given weights shape: {}'.format(
                weights.shape)
        assert (weights.shape[0] == nParticles),\
            'weights must be a (nParticles,). Given weights shape: {}, desired number of particles: {}'.format(
                weights.shape, nParticles)

        assert (len(observation.shape) == 1),\
            'observation must be a 1D numpy.ndarray. Given observation shape: {}'.format(
                observation.shape)
        assert (observation.shape[0] == self.obsDim),\
            'observation must be a (nObservationDim,) numpy.ndarray. Given observation shape: {}, desired observation dimension: {}'.format(
                observation.shape, self.obsDim)

        # Compute the deterministic part of the observation model
        observationPaticules = np.apply_along_axis(
            self.observation_function, 1, particles)
        # According to the parameters of the stochastic part of the observation model
        #(centers= observationPaticules, covariance = observation covariance),
        # what is the likelyhood of the observation
        pdf = multivariateNormalLawPDF(
            observationPaticules, self.observation_covariance, observation)

        # Non-Normalized new weights
        nnweights = weights * pdf
        # Normalized new weights
        new_weights = nnweights / np.sum(nnweights)

        return new_weights

    def _computeEffectiveNParticlesNumber(self, weights):
        """
            Compute the effective number of particles.

            :param weights: old weights
            :type weights: (nParticles,) numpy.ndarray
            :returns: effective number of particles
            :rtype: float
        """

        assert (type(weights) is np.ndarray), 'weights must be a numpy.ndarray'
        assert (len(weights.shape) == 1),\
            'weights must be a 1D numpy.ndarray. Given weights shape: {}'.format(
                weights.shape)

        # Compute the effective number of particles.
        # nEff= ?
        raise NotImplementedError
        return nEff

    def _resample(self, particles, weights, nuNParticles):
        """
            According to the particles weights, sample nuNParticles new particles

            :param particles: particles
            :type particles: (nParticles,nStateDim) numpy.ndarray
            :param weights: weights
            :type weights: (nParticles,) numpy.ndarray
            :param nuNParticles: how many new particles to sample
            :type nuNParticles: int
            :returns: new particles and new weights
            :rtype: ( (nuNParticles,nStateDim) numpy.ndarray, (nuNParticles,) numpy.ndarray )

        """
        assert (type(particles) is np.ndarray), 'particles must be a numpy.ndarray'
        assert (type(weights) is np.ndarray), 'weights must be a numpy.ndarray'

        assert (len(particles.shape) == 2),\
            'particles must be a 2D numpy.ndarray. Given particles shape: {}'.format(
                particles.shape)
        assert (particles.shape[1] == self.stateDim),\
            'particles must be a (nParticles,nStateDim) numpy.ndarray. Given particles shape: {}, desired state dimension: {}'.format(
                particles.shape, self.stateDim)
        nParticles = particles.shape[0]

        assert (len(weights.shape) == 1),\
            'weights must be a 1D numpy.ndarray. Given weights shape {}'.format(
                weights.shape)
        assert (weights.shape[0] == nParticles),\
            'weights must be a (nParticles,). Given weights shape: {}, desired number of particles: {}'.format(
                weights.shape, nParticles)

        # Normalize the weights to have a discrete distribution
        nweights = weights / np.sum(weights)
        # Sample nuNparticles times the discrete distribution
        # Output [3 5 1] means:
        # - 3 occurrences of the first support
        # - 5 occurrences of the second support
        # - 1 occurrences of the third support
        samples = np.random.multinomial(nuNParticles, nweights)

        # Fills the new_particles list with the samples occurrences.
        new_particles = []
        for i in range(len(samples)):
            # Appends particles[i] a number of samples[i] times to new_particles list
            # new_particles+=?
            raise NotImplementedError
        # Converts new_particles back to an numpy.ndarray
        new_particles = np.array(new_particles)

        # new_weights= ?
        raise NotImplementedError
        return (new_particles, new_weights)

    def estimateStatesFromMax(self, particles, weights):
        """
            According to particles and weights from a forward pass, estimates states by the maximum weighted particle

            :param particles: particles
            :type particles: (nObservations+1,nParticles,nStateDim) numpy.ndarray
            :param weights: weights
            :type weights: (nObservations+1,nParticles,) numpy.ndarray
            :returns: estimated states
            :rtype: (nObservations+1,nStateDim) numpy.ndarray

        """

        assert (type(particles) is np.ndarray), 'particles must be a numpy.ndarray'
        assert (type(weights) is np.ndarray), 'weights must be a numpy.ndarray'

        assert (len(particles.shape) == 3),\
            'particles must be a 3D numpy.ndarray. Given particles shape: {}'.format(
                particles.shape)
        assert (particles.shape[2] == self.stateDim),\
            'particles must be a (nObservations+1,nParticles,nStateDim) numpy.ndarray. Given particles shape: {}, desired state dimension: {}'.format(
                particles.shape, self.stateDim)
        nT = particles.shape[0]
        nParticles = particles.shape[1]

        assert (len(weights.shape) == 2),\
            'weights must be a 2D numpy.ndarray. Given weights shape {}'.format(
                weights.shape)
        assert ((weights.shape[0] == nT) and (weights.shape[1] == nParticles)),\
            'weights must be a (nObservations+1,nParticles,). Given weights shape: {}, desired number of observations+1: {} and of particles: {}'.format(
                weights.shape, nT, nParticles)

        states = np.zeros((nT, self.stateDim))
        # For each time step, get the most weighted particle
        pmax = weights.argmax(axis=1)

        for i in range(nT):
            # For each time step, we estimate the current state with the most weighted particle
            # states[i]=?
            raise NotImplementedError

        return states

    def estimateStatesFromMean(self, particles, weights):
        """
            According to particles and weights from a forward pass, estimates states by the weighted mean of particles

            :param particles: particles
            :type particles: (nObservations+1,nParticles,nStateDim) numpy.ndarray
            :param weights: weights
            :type weights: (nObservations+1,nParticles,) numpy.ndarray
            :returns: estimated states
            :rtype: (nObservations+1,nStateDim) numpy.ndarray

        """
        assert (type(particles) is np.ndarray), 'particles must be a numpy.ndarray'
        assert (type(weights) is np.ndarray), 'weights must be a numpy.ndarray'

        assert (len(particles.shape) == 3),\
            'particles must be a 3D numpy.ndarray. Given particles shape: {}'.format(
                particles.shape)
        assert (particles.shape[2] == self.stateDim),\
            'particles must be a (nObservations+1,nParticles,nStateDim) numpy.ndarray. Given particles shape: {}, desired state dimension: {}'.format(
                particles.shape, self.stateDim)
        nT = particles.shape[0]
        nParticles = particles.shape[1]

        assert (len(weights.shape) == 2),\
            'weights must be a 2D numpy.ndarray. Given weights shape {}'.format(
                weights.shape)
        assert ((weights.shape[0] == nT) and (weights.shape[1] == nParticles)),\
            'weights must be a (nObservations+1,nParticles,). Given weights shape: {}, desired number of observations+1: {} and of particles: {}'.format(
                weights.shape, nT, nParticles)

        states = np.zeros((nT, self.stateDim))

        for i in range(nT):
            for j in range(self.stateDim):
                states[i, j] = np.sum(
                    weights[i] * particles[i, :, j]) / np.sum(weights[i])

        return states

    def forward(self, nParticles, observations, resampling_threshold=None):
        """
            Do a forward pass of the particle filter using a simple bootstrap filter.

            :param nParticles: number of particles involved
            :type nParticles: int
            :param observations: observations data
            :type observations: (nObservations,nObservationDim) numpy.ndarray
            :param resampling_threshold: resampling is trigered when the number of effective particles goes below it. Default: nParticles/10.
            :type resampling_threshold: float
            :returns: (particles, weights)
            :rtype: ( (nObservations+1,nParticles,nStateDim) numpy.ndarray , (nObservations+1,nParticles,) numpy.ndarray )

        """
        assert ((type(nParticles) is int) and (nParticles > 0)),\
            'nParticles must be a positive non null int. Given nParticles: {}'.format(
                nParticles)

        assert (type(observations)
                is np.ndarray), 'observations must be a numpy.ndarray'
        assert (len(observations.shape) == 2),\
            'observations must be a 2D numpy.ndarray. Given observations shape: {}'.format(
                observations.shape)
        assert (observations.shape[1] == self.obsDim),\
            'observations must be a (nObservations,nObservationDim). Given observations shape: {}, desired observation dim: {}'.format(
                observations.shape, self.obsDim)
        nT = observations.shape[0]

        assert (((type(resampling_threshold) is float) and (resampling_threshold > 0)) or (resampling_threshold is None)),\
            'resampling_threshold must be a positive non null float or None. Given resampling_threshold {}'.format(
                resampling_threshold)

        if resampling_threshold is None:
            resampling_threshold = float(nParticles) / 10.

        particles = np.zeros((nT + 1, nParticles, self.stateDim))
        weights = np.zeros((nT + 1, nParticles))

        # particles[0]=?
        # weights[0]=?
        raise NotImplementedError

        for k in range(nT):
            # nEff=?
            raise NotImplementedError

            # Performs resampling  if needed
            # if nEff ?:
            #   ?
            # #else :
            #   ?
            raise NotImplementedError

            # compute new particles
            # particles[k+1]=?
            raise NotImplementedError

            # compute new weights
            # weights[k+1]=?
            raise NotImplementedError


        return (particles, weights)
