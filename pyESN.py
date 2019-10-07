"""
Copyright (c) 2015 Clemens Korndörfer
pyESN GitHub repo: https://github.com/cknd/pyESN

Modified by Raytheon BBN Technologies, 2019

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from sklearn.linear_model import Ridge

def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


class ESN():
    """ Echo State Network
    The network has `n_reservoir` nodes, receive input data of length `n_inputs`
    and emit output data of length `n_outputs` as:
        next_states = activation(W_in*input + W*states + W_feedb*output)
        next_output = out_activation(next_states)*W_out

    The input and output data can be scaled before injecting into the network as
        input ==> input_shift + input_scaling*input
        output ==> teacher_shift + teacher_scaling*inverse_out_activation(output)

    The matrices `W_in`, `W` and `W_feedb` are randomly generated.
    `W` has `sparsity` and radius `spectral_radius`.

    The output matrix `W_out` is determined by linear regression against training data.

    Note on matrix sizes:
    For training/testing input data of shape (n_samples,n_depth,n_inputs)
    and ouput data of shape (n_samples,n_outputs):
    W_in: n_reservoir x n_inputs
    W: n_reservoir x n_reservoir
    W_feedb: n_reservoir x n_outputs
    state: n_reservoir
    states: n_samples x n_depth x n_reservoir
    W_out: n_reservoir x n_outputs

    """

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, transient=0, sparsity=0, noise=0.001,
                 input_shift=0, input_scaling=1, include_input=True,
                 teacher_forcing=False, teacher_scaling=1, teacher_shift=0,
                 activation = np.tanh, out_activation=identity, inverse_out_activation=identity,
                 random_state=None):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            transient: number of initial steps to be ignored in fitting
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            include_input: if True, include input data into fitting
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
        """
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs

        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.transient = transient

        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)
        self.include_input = include_input

        self.teacher_forcing = teacher_forcing
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.activation = activation
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation

        self.random_state = random_state
        self._reset()

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.initweights()

    def initweights(self):
        """ Initialize recurrent weights """
        # begin with a random matrix centered around zero
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity)
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius
        self.W = W * (self.spectral_radius / radius)

        # random input weights
        self.W_in = 2*self.random_state_.rand(self.n_reservoir, self.n_inputs) - 1
        # random feedback (teacher forcing) weights
        self.W_feedb = 2*self.random_state_.rand(self.n_reservoir, self.n_outputs) - 1

    def _reset(self):
        self.state = np.zeros(self.n_reservoir)
        self.lastoutput = np.zeros(self.n_outputs)

    def _update(self, input_pattern, output_pattern=0):
        """ Perform one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        input_scaled = self._scale_inputs(input_pattern)
        if self.teacher_forcing:
            output_scaled = self._scale_teacher(output_pattern)
            preactivation = np.dot(self.W, self.state) \
                             + np.dot(self.W_in, input_scaled) \
                             + np.dot(self.W_feedb, output_scaled)
        else:
            preactivation = np.dot(self.W, self.state) \
                             + np.dot(self.W_in, input_scaled)
        self.state = self.activation(preactivation) \
                + self.noise*(self.random_state_.rand(self.n_reservoir) - 0.5)
        return self.state

    def _scale_inputs(self, inputs):
        return np.dot(inputs, np.diag(self.input_scaling)) + self.input_shift

    def _scale_teacher(self, teacher):
        return teacher*self.teacher_scaling + self.teacher_shift

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        return (teacher_scaled - self.teacher_shift) / self.teacher_scaling

    def _reshape(self,data):
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if np.array(data).ndim<2:
            return np.reshape(data,(len(data),-1))
        return np.array(data)

    def process(self,inputs,outputs=0,reset=True):
        """ Receive input data (and output data if teacher_forcing), run through the
        network and record the states of the nodes.

        The input data 'inputs` contains n_samples, each is an array of n_depth (can be 1)
        of n_inputs values. The output data `outputs` is n_samples x n_outputs.
        """
        n_samples = inputs.shape[0]
        n_depth = int(np.prod(inputs.shape) / (n_samples*self.n_inputs))
        if reset:
            self._reset()
        # step the reservoir through the given input,output pairs:
        if n_depth==1:
            states = np.zeros((n_samples, self.n_reservoir))
            if self.teacher_forcing:
                states[0] = self._update(inputs[0],self.lastoutput)
                for n in range(1,n_samples):
                    states[n] = self._update(inputs[n],outputs[n - 1])
            else:
                for n in range(n_samples):
                    states[n] = self._update(inputs[n])
        else:
            """ Note: The usage case for n_depth>1 is different from above
            The network is supposed to process a series of input data of length n_depth
            corresponding to a single output.
            If reset is True: reset between samples.
            No teacher_forcing is supported.
            """
            inputs = inputs.reshape(n_samples,n_depth,self.n_inputs)
            states = np.zeros((n_samples,n_depth, self.n_reservoir))
            for n in range(n_samples):
                if reset:
                    self._reset()
                for i in range(n_depth):
                    states[n,i] = self._update(inputs[n,i])
        return states

    def fit(self, inputs, targets, reset=False, alpha=0):
        """
        Collect the network's reaction to training data, train readout weights.

        Args:
            inputs: array of dimensions (N_training_samples x n_depth x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            alpha: Ridge coefficient

        Returns:
            the network's output on the training data, using the trained weights
        """
        inputs = self._reshape(inputs)
        targets = self._reshape(targets)
        n_samples = inputs.shape[0]
        n_depth = int(np.prod(inputs.shape) / (n_samples*self.n_inputs))
        # Pass the inputs into the network
        states = self.process(inputs,targets,reset=reset)
        # we'll disregard the first few states:
        transient = min(int(inputs.shape[1] / 10), self.transient)
        if n_depth>1:
            states = states[:,transient:,:].reshape(n_samples,-1)
            inputs = inputs.reshape(n_samples,n_depth,self.n_inputs)
            inputs = inputs_scaled[:,transient:,:].reshape(n_samples,-1)
        else:
            states = states[transient:,:]
            inputs = inputs.reshape(n_samples,self.n_inputs)
            inputs = inputs[transient:,:]
        # Include the inputs into the states
        if self.include_input:
            outputs = np.hstack((self.out_activation(states),inputs))
        else:
            outputs = self.out_activation(states)
        # Solve for W_out
        targets = targets[transient:,:]
        ridge_reg = Ridge(alpha=alpha,fit_intercept=False)
        ridge_reg.fit(outputs,targets)
        self.W_out = ridge_reg.coef_.T
        # apply learned weights to the collected states:
        pred_train = np.dot(outputs, self.W_out)
        # remember the last state for later
        self.lastoutput = pred_train[-1,:self.n_outputs]
        return pred_train

    def predict(self, inputs, reset=False):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            inputs: array of dimensions (N_test_samples x x n_depth x n_inputs)
            reset: if False, start the network from the last training state

        Returns:
            Array of output activations
        """
        inputs = self._reshape(inputs)
        n_samples = inputs.shape[0]
        n_depth = int(np.prod(inputs.shape) / (n_samples*self.n_inputs))
        if self.teacher_forcing and n_depth==1:
            if reset:
                self._reset()
            states = np.zeros((n_samples, self.n_reservoir))
            outputs = np.zeros((n_samples,self.n_outputs))
            if self.include_input:
                states[0] = self._update(inputs[0],self.lastoutput)
                outputs[0] = np.dot(np.concatenate([self.out_activation(states[0]),inputs[0]]),
                                                            self.W_out)
                for n in range(1,n_samples):
                    states[n] = self._update(inputs[n],outputs[n-1])
                    outputs[n] = np.dot(np.concatenate([self.out_activation(states[n]),inputs[n]]),
                                                                self.W_out)
            else:
                states[0] = self._update(inputs[0],self.lastoutput)
                outputs[0] = np.dot(self.out_activation(states[0]),self.W_out)
                for n in range(1,n_samples):
                    states[n] = self._update(inputs[n],outputs[n-1])
                    outputs[n] = np.dot(self.out_activation(states[n]),self.W_out)
        else:
            states = self.out_activation(self.process(inputs,reset=reset))
            # Include the inputs into the states
            if self.include_input:
                extended_states = np.hstack((states.reshape(n_samples,-1),inputs.reshape(n_samples,-1)))
            else:
                extended_states = states.reshape(n_samples,-1)
            outputs = np.dot(extended_states,self.W_out)
        self.lastoutput = outputs[-1,:self.n_outputs]
        return outputs
