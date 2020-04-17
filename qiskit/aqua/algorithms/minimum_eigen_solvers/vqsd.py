# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Variational Quantum Eigendecomposition algorithm """

from typing import Optional, List, Callable, Union
from time import time
from itertools import product
import logging
import functools
import warnings
import numpy as np

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils.backend_utils import is_aer_provider
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.components.optimizers import Optimizer, SLSQP
from qiskit.aqua.components.variational_forms import VariationalForm, RY
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.utils.validation import validate_min, validate_range
from qiskit.aqua.utils.run_circuits import find_regs_by_name
from qiskit.quantum_info.states import Statevector
from qiskit.quantum_info import DensityMatrix
# TODO : Fix temporary solution for path related issues
from vq_algorithm import VQAlgorithm, VQResult
from minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult

logger = logging.getLogger(__name__)

# disable check for var_forms, optimizer setter because of pylint bug
# pylint: disable=no-member


class VQSD(VQAlgorithm, MinimumEigensolver):
    r"""
    The Variational Quantum Eigensolver algorithm.
    `VQE <https://arxiv.org/abs/1304.3061>`__ is a hybrid algorithm that uses a
    variational technique and interleaves quantum and classical computations in order to find
    the minimum eigenvalue of the Hamiltonian :math:`H` of a given system.
    An instance of VQE requires defining two algorithmic sub-components:
    a trial state (ansatz) from Aqua's :mod:`~qiskit.aqua.components.variational_forms`, and one
    of the classical :mod:`~qiskit.aqua.components.optimizers`. The ansatz is varied, via its set
    of parameters, by the optimizer, such that it works towards a state, as determined by the
    parameters applied to the variational form, that will result in the minimum expectation value
    being measured of the input operator (Hamiltonian).
    An optional array of parameter values, via the *initial_point*, may be provided as the
    starting point for the search of the minimum eigenvalue. This feature is particularly useful
    such as when there are reasons to believe that the solution point is close to a particular
    point.  As an example, when building the dissociation profile of a molecule,
    it is likely that using the previous computed optimal solution as the starting
    initial point for the next interatomic distance is going to reduce the number of iterations
    necessary for the variational algorithm to converge.  Aqua provides an
    `initial point tutorial <https://github.com/Qiskit/qiskit-tutorials-community/blob/master
    /chemistry/h2_vqe_initial_point.ipynb>`__ detailing this use case.
    The length of the *initial_point* list value must match the number of the parameters
    expected by the variational form being used. If the *initial_point* is left at the default
    of ``None``, then VQE will look to the variational form for a preferred value, based on its
    given initial state. If the variational form returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the variational form provides ``None`` as the lower bound, then VQE
    will default it to :math:`-2\pi`; similarly, if the variational form returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.
    """
    def __init__(self,
                 initial_state: InitialState,
                 operator: Optional[BaseOperator] = None,
                 q: Optional[float] = 0,
                 num_ancillae: Optional[int] = 0,
                 var_form: Optional[VariationalForm] = None,
                 optimizer: Optional[Optimizer] = None,
                 initial_point: Optional[np.ndarray] = None,
                 max_evals_grouped: int = 1,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        """
        Constructor.

        Args:
            initial_state (InitialState): The state to be diagonalized
            operator (BaseOperator): The density matrix of the
            initial state
            q (int): Free parameter that ones to tailer the VQSD method
            num_ancillae (int): The number of ancillae qubits if the initial
            state is a mixed state
            var_form: A parameterized variational form (ansatz).
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                variational form, the evaluated global cost, local cost and
                weighted cost
            quantum_instance: Quantum Instance or Backend
        """

        validate_min('max_evals_grouped', max_evals_grouped, 1)
        validate_range('num_ancillae', num_ancillae, 0, initial_state._num_qubits - 1)
        validate_range('q', q, 0.0, 1.0)

        if var_form is None:
            # TODO after ansatz refactor num qubits can be set later so we do not have to have
            #      an operator to create a default
            if initial_state is not None:
                var_form = RY(initial_state._num_qubits -
                              num_ancillae)

        if optimizer is None:
            optimizer = SLSQP()

        if operator is None:
            initial_state_vector = initial_state.construct_circuit(mode='vector')
            mat = np.outer(initial_state_vector, np.conj(initial_state_vector))
            operator = DensityMatrix(mat)

        # TODO after ansatz refactor we may still not be able to do this
        # if num qubits is not set on var form
        if initial_point is None and var_form is not None:
            initial_point = var_form.preferred_init_points

        self._max_evals_grouped = max_evals_grouped

        super().__init__(var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=self._cost_evaluation,
                         initial_point=initial_point,
                         quantum_instance=quantum_instance)

        self._callback = callback
        self._use_simulator_snapshot_mode = None
        self._ret = None
        self._eval_time = None
        self._eval_count = -1

        logger.info(self.print_settings())
        self._var_form_params = None
        if self.var_form is not None:
            self._var_form_params = ParameterVector('θ', self.var_form.num_parameters)
        self._parameterized_circuits = None

        self._initial_state = initial_state
        self._q = q
        self._num_ancillae = num_ancillae
        self._num_working_qubits = initial_state._num_qubits - num_ancillae
        self._operator = operator
        self.initial_state = initial_state

        # TODO : Verify that if the ancillae qubits form an orthonormal basis

        # Compute state purity
        if self._num_ancillae > 0:
            # pylint: disable=import-outside-toplevel
            from qiskit.quantum_info import purity, partial_trace

            rho = self._operator.data
            ancillae_idx = list(set(range(self._initial_state._num_qubits)) -
                                set(range(self._num_ancillae)))
            self._operator = partial_trace(rho, ancillae_idx)
            self._purity = purity(self._operator)
        else:
            self._purity = 1.0

    @property
    def operator(self) -> Optional[BaseOperator]:
        """ Returns operator """
        raise TypeError("Operators not supported")

    @operator.setter
    def operator(self, operator: BaseOperator) -> None:
        """ Set operator """
        raise TypeError("Operators not supported")

    @property
    def aux_operators(self) -> List[BaseOperator]:
        """ Returns aux operators """
        raise TypeError('Aux_operators not supported.')

    @aux_operators.setter
    def aux_operators(self, aux_operators: List[BaseOperator]) -> None:
        """ Set aux operators """
        raise TypeError('Aux_operators not supported.')

    @VQAlgorithm.var_form.setter
    def var_form(self, var_form: VariationalForm):
        """ Sets variational form """
        VQAlgorithm.var_form.fset(self, var_form)
        self._var_form_params = ParameterVector('θ', var_form.num_parameters)
        if self.initial_point is None:
            self.initial_point = var_form.preferred_init_points
        self._check_initial_state_varform()

    def _check_initial_state_varform(self):
        if self.initial_state is not None and self.var_form is not None:
            if self.initial_state._num_qubits != self.var_form.num_qubits:
                # TODO After Ansatz update we should be able to set in the
                #      number of qubits to var form. Important since use by
                #      application stack of VQE the user may be able to set
                #      a var form but not know num_qubits. Whether any smarter
                #      settings could be optionally done by VQE e.g adjust depth
                #      is TBD. Also this auto adjusting might not be reasonable for
                #      instance UCCSD where its parameterization is much closer to
                #      the specific problem and hence to the operator
                raise AquaError("Variational form num qubits does not match "
                                "initial_state")

    @VQAlgorithm.optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        """ Sets optimizer """
        super().optimizer = optimizer
        if optimizer is not None:
            optimizer.set_max_evals_grouped(self._max_evals_grouped)

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    def print_settings(self):
        """
        Preparing the setting of VQSD into a string.
        Returns:
            str: the formatted setting of VQSD
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.__class__.__name__)
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        if self._var_form is not None:
            ret += "{}".format(self._var_form.setting)
        else:
            ret += 'var_form has not been set'
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

    def compute_minimum_eigenvalue(
            self, operator: Optional[BaseOperator] = None,
            aux_operators: Optional[List[BaseOperator]] = None) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        return self._run()

    def supports_aux_operators(self) -> bool:
        return False

    def _run(self) -> 'VQSDResult':
        """
        Run the algorithm to compute the minimum eigenvalue.
        Returns:
            dict: Dictionary of results
        Raises:
            AquaError: wrong setting of operator and backend.
        """
        if self.initial_state is None:
            raise AquaError("Initial state was never provided")

        self._initial_state = self.initial_state
        self._use_simulator_snapshot_mode = (
            is_aer_provider(self._quantum_instance.backend)
            and self._quantum_instance.run_config.shots == 1
            and not self._quantum_instance.noise_config)

        self._quantum_instance.circuit_summary = True

        self._eval_count = 0
        vqresult = self.find_minimum(initial_point=self.initial_point,
                                     var_form=self.var_form,
                                     cost_fn=self._cost_evaluation,
                                     optimizer=self.optimizer)

        # TODO remove all former dictionary logic
        self._ret = {}
        self._ret['num_optimizer_evals'] = vqresult.optimizer_evals
        self._ret['min_val'] = vqresult.optimal_value
        self._ret['opt_params'] = vqresult.optimal_point
        self._ret['eval_time'] = vqresult.optimizer_time

        if self._ret['num_optimizer_evals'] is not None and \
                self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        self._ret['eigvals'] = self.get_optimal_eigenvalues()
        self._ret['eigvecs'] = self.get_optimal_vector()

        result = VQSDResult()
        result.combine(vqresult)
        result.eigenvalues = self.get_optimal_eigenvalues()
        result.eigenstates = self.get_optimal_vector()
        result.cost_function_evals = self._eval_count

        self.cleanup_parameterized_circuits()
        return result

    def get_optimal_cost(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the "
                            "algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the "
                            "algorithm to find optimal params.")

        num_qubits = self._initial_state._num_qubits
        num_working_qubits = self._num_working_qubits
        initial_state_circuit = self._initial_state.construct_circuit(mode='circuit')
        opt_ansatz_circuit = self._var_form.construct_circuit(self._ret['opt_params'])

        qreg = QuantumRegister(num_qubits, name='q')
        circuit = QuantumCircuit(qreg)
        circuit.append(initial_state_circuit, qreg)
        circuit.append(opt_ansatz_circuit, qreg[:num_working_qubits])
        return circuit

    def get_optimal_vector(self):
        """
        Construct the eigenvector estimates of the now approximately
        diagonalized density matrix

        Returns:
            numpy.ndarray: An array of the eigenvector estimates
        """
        # pylint: disable=import-outside-toplevel
        from qiskit import execute, Aer

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal eigenvectors before running the "
                            "algorithm to find optimal params.")

        # TODO: Find a way to sort the eigenvectors according to their
        # eigenvalues
        num_working_qubits = self._num_working_qubits
        opt_ansatz_circuit = self._var_form.construct_circuit(self._ret['opt_params'])
        keys = list(product(['0', '1'],
                                      repeat=num_working_qubits))

        eigvecs = np.zeros((len(keys), 2**num_working_qubits), dtype='complex')
        j = 0
        for key in keys:
            qreg = QuantumRegister(num_working_qubits)
            circuit = QuantumCircuit(qreg)
            for i in range(num_working_qubits):
                if key[i] == '1':
                    circuit.x(i)
            circuit.append(opt_ansatz_circuit.inverse(), qreg)

            backend = Aer.get_backend('statevector_simulator')
            eigvec = execute(circuit, backend).result().get_statevector(circuit)
            eigvecs[j] = eigvec
            j += 1
        return eigvecs

    def get_optimal_eigenvalues(self):
        """
        Computes the eigenvalue estimates of the initial state
        """
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the "
                            "algorithm to find optimal params.")

        # TODO : Find better way to do this
        num_qubits = self._initial_state._num_qubits
        num_working_qubits = self._num_working_qubits
        circuit = self.get_optimal_circuit()
        counts = None

        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(circuit)
            state_vec = Statevector(ret.get_statevector(circuit))
            counts = state_vec.probabilities_dict()
        else:
            c = ClassicalRegister(circuit.width(), name='c')
            q = find_regs_by_name(circuit, 'q')
            circuit.add_register(c)
            circuit.barrier(q)
            circuit.measure(q[:num_working_qubits], c[:num_working_qubits])
            ret = self._quantum_instance.execute(circuit)
            counts = ret.get_counts(circuit)

        all_keys = list(product(['0', '1'], repeat=num_qubits))
        eigvals = np.zeros(len(all_keys))
        keys = counts.keys()
        num_shots = sum(counts.values())
        j = 0
        for key in all_keys:
            k = ''.join(key)
            if k in keys:
                eigvals[j] = counts[k] / num_shots
                j += 1
        # sort eigenvalues in descending order
        idx = np.argsort(-eigvals)
        return eigvals[idx[:2**num_working_qubits]]

    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before "
                            "running the algorithm.")
        return self._ret['opt_params']

    def _pdip_test(self, parameter, dip_qubit_idx,
                   swap_qubit_idx,
                   circuit_name_prefix=''):
        """
        Performs the PDIP Test on a circuit composed of the initial state and
        a parameterized ansatz.

        Args:
            parameter (numpy.ndarray): parameters for variational form.
            dip_qubit_idx (numpy.ndarray): indices of qubits on which the DIP
                Test is performed
            swap_qubit_idx (numpy.ndarray): indices of qubits on which the SWAP
                Test is performed
            circuit_name_prefix (str, optional): a prefixe of circuit name
        Returns:
            QuantumCircuit: Generated circuit
        """
        initial_state = self._initial_state
        num_qubits = initial_state._num_qubits
        num_working_qubits = self._num_working_qubits

        qreg = QuantumRegister(2 * num_qubits, name='pdip_qreg')
        circuit_name = "{}_pdip_test".format(circuit_name_prefix)
        circuit = QuantumCircuit(qreg, name=circuit_name)
        pdip_circuit = VQSD._pdip_circuit(num_working_qubits,
                                          dip_qubit_idx, swap_qubit_idx)

        initial_state_circuit = initial_state.construct_circuit(mode="circuit")
        ansatz_circuit = self._var_form.construct_circuit(parameter)

        # two copies of initial state
        circuit.append(initial_state_circuit, qreg[:num_qubits])
        circuit.append(initial_state_circuit, qreg[num_qubits:])

        # two copies of ansatz
        circuit.append(ansatz_circuit, qreg[:num_working_qubits])
        circuit.append(ansatz_circuit, qreg[num_qubits:num_qubits + num_working_qubits])

        circuit.append(pdip_circuit, qreg[:num_working_qubits] +
                       qreg[num_qubits:num_qubits+num_working_qubits])
        return circuit

    def _dip_test(self, parameter, circuit_name_prefix=''):
        """
        Performs the PDIP Test on a circuit composed of the initial state and
        a parameterized ansatz.
        Args:

            parameter (numpy.ndarray): parameters for variational form.
            circuit_name_prefix (str, optional): a prefix of circuit name
        Returns:
            QuantumCircuit: Generated circuit
        """
        num_qubits = self._initial_state._num_qubits
        num_working_qubits = self._num_working_qubits

        qreg = QuantumRegister(2 * num_qubits, name="dip_qreg")
        circuit_name = "{}_dip_test".format(circuit_name_prefix)
        circuit = QuantumCircuit(qreg, name=circuit_name)

        dip_circuit = VQSD._dip_circuit(self._num_working_qubits)
        initial_state_circuit = self._initial_state.construct_circuit(mode="circuit")
        ansatz_circuit = self._var_form.construct_circuit(parameter)

        # two copies of initial state
        circuit.append(initial_state_circuit, qreg[:num_qubits])
        circuit.append(initial_state_circuit, qreg[num_qubits:])

        # two copies of ansatz
        circuit.append(ansatz_circuit, qreg[:num_working_qubits])
        circuit.append(ansatz_circuit, qreg[num_qubits:num_qubits + num_working_qubits])

        circuit.append(dip_circuit, qreg[:num_working_qubits] +
                       qreg[num_qubits:num_qubits + num_working_qubits])
        return circuit

    def construct_circuit(self, parameter, statevector_mode=False,
                          use_simulator_snapshot_mode=False,
                          circuit_name_prefix=''):
        """
        Generate the circuits.
        Args:
            parameter (numpy.ndarray): parameters for variational form.
            statevector_mode (bool, optional): indicate which type of simulator are going to use.
            use_simulator_snapshot_mode (bool, optional): is backend from AerProvider,
                            if True and mode is paulis, single circuit is generated.
            circuit_name_prefix (str, optional): a prefix of circuit name
        Returns:
            list[QuantumCircuit]: the generated circuits with Hamiltonian.
        Raises:
            AquaError: Circuit cannot be created if an operator has not been provided
        """
        if self._initial_state is None:
            raise AquaError("Initial state was never provided")

        num_working_qubits = self._num_working_qubits

        circuits = []
        dip_test = self._dip_test(parameter, circuit_name_prefix=circuit_name_prefix)

        if not statevector_mode:
            creg = ClassicalRegister(dip_test.width(), name="dip_creg")
            qreg = find_regs_by_name(dip_test, 'dip_qreg')
            dip_test.add_register(creg)
            dip_test.barrier(qreg)
            dip_test.measure(qreg[:num_working_qubits], creg[:num_working_qubits])
        circuits.append(dip_test)

        for i in range(self._num_working_qubits):
            # get relevant qubit indices
            dip_qubit_idx = [i]
            swap_qubit_idx = list(set(range(num_working_qubits)) - set(dip_qubit_idx))
            swap_qubit_idx += [i + num_working_qubits for i in swap_qubit_idx]
            dip_qubit_idx += [i + num_working_qubits for i in dip_qubit_idx]
            pdip_test = self._pdip_test(parameter,
                                        dip_qubit_idx, swap_qubit_idx,
                                        circuit_name_prefix=circuit_name_prefix)

            if not statevector_mode:
                creg = ClassicalRegister(pdip_test.width(), name="dip_creg")
                qreg = find_regs_by_name(pdip_test, 'pdip_qreg')
                pdip_test.add_register(creg)
                pdip_test.barrier(qreg)
                pdip_test.measure(qreg[swap_qubit_idx + dip_qubit_idx],
                                  creg[swap_qubit_idx + dip_qubit_idx])
            circuits.append(pdip_test)
        return circuits

    # This is the objective function to be passed to the optimizer that is
    # used for evaluation
    def _cost_evaluation(self, parameters):
        """
        Evaluate cost function at given parameters for the variational form.
        Args:
            parameters (numpy.ndarray): parameters for variational form.
        Returns:
            Union(float, list[float]): cost objective function value of each parameter.
        """
        num_parameter_sets = len(parameters) // self._var_form.num_parameters
        parameter_sets = np.split(parameters, num_parameter_sets)
        global_objs = []
        local_objs = []
        weighted_objs = []

        def _build_parameterized_circuits():
            if self._var_form.support_parameterized_circuit and \
                    self._parameterized_circuits is not None:
                parameterized_circuits = self.construct_circuit(
                    self._var_form_params,
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=self._use_simulator_snapshot_mode)

                self._parameterized_circuits = \
                    self._quantum_instance.transpile(parameterized_circuits)

        _build_parameterized_circuits()
        circuits = []
        # binding parameters here since the circuits had been transpiled
        if self._parameterized_circuits is not None:
            for idx, parameter in enumerate(parameter_sets):
                curr_param = {self._var_form_params: parameter}
                for qc in self._parameterized_circuits:
                    tmp = qc.bind_parameters(curr_param)
                    tmp.name = str(idx) + '_' + tmp.name
                    circuits.append(tmp)
            to_be_simulated_circuits = circuits
        else:
            for idx, parameter in enumerate(parameter_sets):
                circuit = self.construct_circuit(
                    parameter,
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                    circuit_name_prefix=str(idx))
                circuits.append(circuit)
            to_be_simulated_circuits = functools.reduce(lambda x, y:
                                                        x + y, circuits)

        start_time = time()
        result = self._quantum_instance.execute(to_be_simulated_circuits,
                                                self._parameterized_circuits
                                                is not None)

        for idx, _ in enumerate(parameter_sets):
            # Evaluate with result
            circuits = list(filter(lambda circ:
                                   circ.name.startswith('{}_'.format(idx)),
                                   to_be_simulated_circuits))

            # DIP Test circuits
            dip_test_circuit = circuits[0]

            # PDIP circuits
            pdip_test_circuits = circuits[1:]

            # evaluate with results
            dip_test_counts = None
            pdip_test_counts = None
            if self._quantum_instance.is_statevector:
                np_state_vec = result.get_statevector(dip_test_circuit)
                dip_test_state_vec = Statevector(np_state_vec)
                dip_test_counts = dip_test_state_vec.probabilities_dict()
                pdip_test_counts = \
                    [Statevector(result.get_statevector(circ)).probabilities_dict()
                     for circ in pdip_test_circuits]
            else:
                dip_test_counts = result.get_counts(dip_test_circuit)
                pdip_test_counts = [result.get_counts(circ) for circ in
                                    pdip_test_circuits]

            obj_global = self._dip_test_post_process(dip_test_counts)
            obj_local = self._pdip_test_post_process(pdip_test_counts)
            obj_weigthed = self._q * obj_global + (1.0 - self._q) * obj_local
            end_time = time()

            global_objs.append(np.real(obj_global))
            local_objs.append(np.real(obj_local))
            weighted_objs.append(np.real(obj_weigthed))

            self._eval_count += 1
            if self._callback is not None:
                self._callback(self._eval_count, parameter_sets[idx],
                               np.real(obj_global), np.real(obj_local),
                               np.real(obj_weigthed))

            # If there is more than one parameter set then the calculation
            # of the evaluation time has to be done more carefully,
            # therefore we do not calculate it
            if len(parameter_sets) == 1:
                logger.info('Cost function evaluation %s '
                            'returned %s - %.5f (ms)',
                            self._eval_count,
                            np.real(obj_weigthed),
                            (end_time - start_time) * 1000)
            else:
                logger.info('Cost function evaluation %s returned %s',
                            self._eval_count,
                            np.real(obj_weigthed))
            return weighted_objs if len(weighted_objs) > 1 \
                                    else weighted_objs[0]

    def _dip_test_post_process(self, counts):
        """
        Computes the objective function of the resolved circuit as computed
        by the DIP Test

        Args:
            counts (dict) : dictionary of counts from the DIP Test
        Returns:
            float: C1 from the paper
        Raises:
            AquaError: probabilities are always positive and less than
            or equal to 1.0
        """

        initial_state_purity = self._purity
        num_shots = self._quantum_instance._run_config.shots
        num_working_qubits = self._num_working_qubits

        all_zero_outcome_key = '0' * num_working_qubits
        all_zero_outcomes = {key: value for (key, value) in counts.items()
                             if key[-num_working_qubits:] ==
                             all_zero_outcome_key}

        prob = sum(all_zero_outcomes.values()) / num_shots
        if not 0 <= prob <= 1:
            raise AquaError('Invalid probability')

        return initial_state_purity - prob

    def _pdip_test_post_process(self, counts):
        """
        Computes the objective function of the resolved circuit as computed by
        the Partial DIP Test.

        Args:
            counts (dict): dictionary of counts
        Returns:
            float: C2 from the paper
        Raises:
            AquaError: probabilities are always positive and less than
            or equal to 1.0
        """

        initial_state_purity = self._purity
        num_shots = self._quantum_instance._run_config.shots
        num_qubits = self._initial_state._num_qubits
        num_working_qubits = self._num_working_qubits

        def get_pdip_zero_outcome(c, j):
            """Gets a dictionary of counts where the DIP test outcomes
                were zero"""
            new_dict = {}
            for k in c.keys():
                if k[-1-j] == '0':
                    if k in new_dict.keys():
                        new_dict[k] += c[k]
                    else:
                        new_dict[k] = c[k]
            return new_dict

        def _state_overlap_post_process(counts, skip_idx=None):
            """
            State overlap post processing

            Args:
                counts (dict): dictionary of counts
                num_qubits (int): number of qubits
                skip_idx (list, optional): qubit indices to skip
            Returns:
                int: value of state overlap
            """

            # number of qubits and number of shots
            keys, vals = counts.keys(), counts.values()
            num_qubits, num_shots = len(list(keys)[0]), sum(list(vals))

            # sanity check
            if not num_qubits % 2 == 0:
                raise AquaError("Input is not a valid shape.")

            # Compute expectation value of controlled Z operation
            overlap = 0.0
            shift = num_qubits // 2
            skip_idx = skip_idx if skip_idx is not None else []
            for key in keys:
                parity = 1
                # skip dip qubits and ancillary qubits indices.
                pairs = [0 if ii in skip_idx else
                         int(key[ii]) and int(key[ii + shift])
                         for ii in range(shift)]
                for pair in pairs:
                    parity *= (-1)**pair
                overlap += parity * counts[key]
            return overlap / num_shots

        scaled_overlap = 0.0
        for idx, cur in enumerate(counts):
            pdip_count = get_pdip_zero_outcome(cur, idx)
            overlap = _state_overlap_post_process(pdip_count, skip_idx=[idx])
            # probability of all zero outcome
            all_zero_outcome = '0' * 2 * num_qubits
            prob = pdip_count[all_zero_outcome] / num_shots \
                if all_zero_outcome in pdip_count.keys() else 0
            if not 0 <= prob <= 1:
                raise AquaError('Invalid probability')
            overlap *= prob
            scaled_overlap += overlap
        return initial_state_purity - scaled_overlap / num_working_qubits

    @staticmethod
    def _dip_circuit(num_qubits):
        """
        Construct DIP Test over two copies of the initial state

        Args:
            num_qubits (int): Number of qubits in a single copy of the
            initial state
        Returns:
            QuantumCircuit: Generated circuit
        """

        qreg = QuantumRegister(2 * num_qubits)
        circuit = QuantumCircuit(qreg)
        for i in range(num_qubits):
            circuit.cx(i + num_qubits, i)
        return circuit

    @staticmethod
    def _pdip_circuit(num_qubits, dip_qubit_idx, swap_qubit_idx):
        """
        Implements the Partial DIP Test circuit over two copies of the
        initial state

        Args:
            num_qubits (int): number of qubits in a single copy of the
                initial state
            dip_qubit_idx (list): list of qubit indices (j in the paper)
                to do the dip test on
            swap_qubit_idx (list): list of qubit indices (j in the paper)
                to do the swap test on.
        Returns:
            QuantumCircuit: Generated circuit
        """

        def swap_circuit(num_qubits):
            """
            Construct Destructive Swap Test over two copies of the initial state

            Args:
                num_qubits (int): Number of qubits in a single copy of the
                    initial state
            Returns:
                QuantumCircuit: Generated circuit
            """

            qreg = QuantumRegister(2 * num_qubits)
            circuit = QuantumCircuit(qreg)
            for i in range(num_qubits):
                circuit.cx(i + num_qubits, i)
                circuit.h(i + num_qubits)
            return circuit

        qreg = QuantumRegister(2 * num_qubits)
        pdip_circuit = QuantumCircuit(qreg)

        nswap = len(swap_qubit_idx) // 2
        ndip = len(dip_qubit_idx) // 2

        if nswap > 0:
            swap_circuit = swap_circuit(nswap)
            pdip_circuit.append(swap_circuit, swap_qubit_idx)
        dip_circuit = VQSD._dip_circuit(ndip)
        pdip_circuit.append(dip_circuit, dip_qubit_idx)
        return pdip_circuit


class VQSDResult(VQResult, MinimumEigensolverResult):
    """ VQSD Result."""

    @property
    def cost_function_evals(self) -> int:
        """ Returns number of cost optimizer evaluations """
        return self.get('cost_function_evals')

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """ Sets number of cost function evaluations """
        self.data['cost_function_evals'] = value

    def __getitem__(self, key: object) -> object:
        if key == 'eval_count':
            warnings.warn('eval_count deprecated, use \
                          cost_function_evals property.',
                          DeprecationWarning)
            return super().__getitem__('cost_function_evals')

        try:
            return VQResult.__getitem__(self, key)
        except KeyError:
            return MinimumEigensolverResult.__getitem__(self, key)
