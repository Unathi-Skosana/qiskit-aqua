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
import logging
import functools
import itertools
import warnings
import numpy as np

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_aer_provider)
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.components.optimizers import Optimizer, SLSQP
from qiskit.aqua.components.variational_forms import VariationalForm, RY
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.utils.validation import validate_min, validate_range
from qiskit.quantum_info import DensityMatrix

# TODO : Fix temporary solution for path related issues
from vq_algorithm import VQAlgorithm, VQResult
from minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult

logger = logging.getLogger(__name__)


class VQSD(VQAlgorithm, MinimumEigensolver):
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
        self._num_working_qubits =  initial_state._num_qubits - num_ancillae
        self._operator = operator

        self.initial_state = initial_state

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
                raise AquaError("Variational form num qubits does not match \
                                initial_state")

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
        pass

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
            raise AquaError("Operator was never provided")

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
        result.eigenvalue = self.get_optimal_eigenvalues()
        result.eigenstate = self.get_optimal_vector()
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
        initial_state_circ = self._initial_state.construct_circuit(mode='circuit')
        opt_ansatz_circ = self._var_form.construct_circuit(self._ret['opt_params'])

        qr = QuantumRegister(num_qubits, name='q')
        qc = QuantumCircuit(qr)

        qc.append(initial_state_circ, qr)
        qc.append(opt_ansatz_circ, qr[:num_working_qubits])

        return qc

    def get_optimal_vector(self):
        # pylint: disable=import-outside-toplevel
        from qiskit import execute, BasicAer

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal eigenvectors before running the "
                            "algorithm to find optimal params.")

        num_working_qubits = self._num_working_qubits
        opt_ansatz_circ = self._var_form.construct_circuit(self._ret['opt_params'])

        eigvecs = []
        keys = list(itertools.product(['0', '1'],
                                      repeat=num_working_qubits))

        for key in keys:
            qr = QuantumRegister(num_working_qubits)
            qc = QuantumCircuit(qr)
            for i in range(num_working_qubits):
                if key[i] == '1':
                    qc.x(i)
            qc.append(opt_ansatz_circ.inverse(), qr)

            backend = BasicAer.get_backend('statevector_simulator')
            result = execute(qc, backend=backend).result()
            eigvec = result.get_statevector(qc)
            eigvecs.append(eigvec)

        return np.array(eigvecs)

    def get_optimal_eigenvalues(self):
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the "
                            "algorithm to find optimal params.")

        num_qubits = self._initial_state._num_qubits
        num_working_qubits = self._num_working_qubits

        qc = self.get_optimal_circuit()

        counts = None
        eigvals = []

        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            counts = ret.get_statevector(qc).to_counts()
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q[:num_working_qubits], c[:num_working_qubits])
            ret = self._quantum_instance.execute(qc)
            counts = ret.get_counts(qc)

        all_keys = list(itertools.product(['0', '1'],
                                          repeat=num_qubits))
        keys = counts.keys()
        num_shots = sum(list(counts.values()))

        for tup in all_keys:
            key = ''.join(tup)
            if key not in keys:
                eigvals.append(0.0)
            else:
                eigvals.append(counts[key] / num_shots)

        return np.array(eigvals)

    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']

    def _pdip_test(self, parameter, dip_idx,
                   statevector_mode=False,
                   use_simulator_snapshot_mode=False,
                   circuit_name_prefix=''):
        num_qubits = self._initial_state._num_qubits
        num_working_qubits = self._num_working_qubits
        first_copy_qr = QuantumRegister(num_qubits)
        second_copy_qr = QuantumRegister(num_qubits)
        full_cr = ClassicalRegister(2 * num_qubits)

        qc_name = "{}_pdip_test".format(circuit_name_prefix)

        qc = QuantumCircuit(first_copy_qr, second_copy_qr,
                            full_cr, name=qc_name)

        pdip_circ = VQSD._pdip_circuit(self._num_working_qubits, dip_idx)
        initial_state_circ = self._initial_state.construct_circuit(mode="circuit")
        ansatz_circ = self._var_form.construct_circuit(parameter)

        # two copies of initial state
        qc.append(initial_state_circ, first_copy_qr)
        qc.append(initial_state_circ, second_copy_qr)

        # two copies of ansatz
        qc.append(ansatz_circ, first_copy_qr[:num_working_qubits])
        qc.append(ansatz_circ, second_copy_qr[:num_working_qubits])

        qc.append(pdip_circ, first_copy_qr[:num_working_qubits] +
                  second_copy_qr[:num_working_qubits], full_cr[:num_working_qubits] +
                  full_cr[num_qubits:num_qubits + num_working_qubits])
        return qc

    def _dip_test(self, parameter, statevector_mode=False,
                  use_simulator_snapshot_mode=False,
                  circuit_name_prefix=''):
        num_qubits = self._initial_state._num_qubits
        num_working_qubits = self._num_working_qubits
        first_copy_qr = QuantumRegister(num_qubits)
        second_copy_qr = QuantumRegister(num_qubits)
        full_cr = ClassicalRegister(2 * num_qubits)

        qc_name = "{}_dip_test".format(circuit_name_prefix)

        qc = QuantumCircuit(first_copy_qr, second_copy_qr,
                            full_cr, name=qc_name)


        dip_circ = VQSD._dip_circuit(self._num_working_qubits)
        initial_state_circ = self._initial_state.construct_circuit(mode="circuit")
        ansatz_circ = self._var_form.construct_circuit(parameter)

        # two copies of initial state
        qc.append(initial_state_circ, first_copy_qr)
        qc.append(initial_state_circ, second_copy_qr)

        # two copies of ansatz
        qc.append(ansatz_circ, first_copy_qr[:num_working_qubits])
        qc.append(ansatz_circ, second_copy_qr[:num_working_qubits])

        qc.append(dip_circ, first_copy_qr[:num_working_qubits] +
                  second_copy_qr[:num_working_qubits], full_cr[:num_working_qubits] +
                  full_cr[num_qubits:num_qubits + num_working_qubits])
        return qc


    def construct_circuit(self, parameter, statevector_mode=False,
                          use_simulator_snapshot_mode=False, circuit_name_prefix=''):
        """Generate the circuits.
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

        circuits = []

        dip_test = self._dip_test(parameter, statevector_mode=statevector_mode,
                                  use_simulator_snapshot_mode=use_simulator_snapshot_mode,
                                  circuit_name_prefix=circuit_name_prefix)
        circuits.append(dip_test)

        for i in range(self._num_working_qubits):
            pdip_test = self._pdip_test(parameter,
                                        [i], statevector_mode=statevector_mode,
                                        circuit_name_prefix=circuit_name_prefix)
            circuits.append(pdip_test)
        return circuits

    def _cost_evaluation(self, parameters):
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
                                                self._parameterized_circuits is not None)

        for idx, _ in enumerate(parameter_sets):
            circuits = list(filter(lambda circ:
                                   circ.name.startswith('{}_'.format(idx)),
                                   to_be_simulated_circuits))

            # evaluate with results
            dip_test_circuit = circuits[0]
            pdip_test_circuits = circuits[1:]

            dip_test_counts = result.get_counts(dip_test_circuit)
            pdip_test_counts = list(map(lambda x: result.get_counts(x),
                                        pdip_test_circuits))

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

            # If there is more than one parameter set then the calculation of the
            # evaluation time has to be done more carefully,
            # therefore we do not calculate it
            if len(parameter_sets) == 1:
                logger.info('Cost function evaluation %s returned %s - %.5f (ms)',
                            self._eval_count,
                            np.real(obj_weigthed),
                            (end_time - start_time) * 1000)
            else:
                logger.info('Cost function evaluation %s returned %s',
                            self._eval_count,
                            np.real(obj_weigthed))

            return weighted_objs if len(weighted_objs) > 1 else weighted_objs[0]

    def _dip_test_post_process(self, counts):
        """Returns the objective function of the resolved circuit as computed
            by the DIP Test
        """

        initial_state_purity = self._purity
        num_shots = self._quantum_instance._run_config.shots
        num_working_qubits = self._num_working_qubits

        prob = sum({key: value for (key, value) in counts.items() if
                    key[-1-num_working_qubits:] == '00'}.values()) / num_shots

        if not 0 <= prob <= 1:
            raise AquaError('Invalid probability')

        return initial_state_purity - prob

    def _pdip_test_post_process(self, counts):
        """Returns the objective function of the resolved circuit as computed by
            the PDIP Test.
        """

        initial_state_purity = self._purity
        num_shots = self._quantum_instance._run_config.shots
        num_qubits = self._initial_state._num_qubits
        num_working_qubits = self._num_working_qubits

        def get_pdip_zero_outcome(c, j):
            new_dict = {}
            for k in c.keys():
                if k[-1-j] == '0':
                    if k in new_dict.keys():
                        new_dict[k] += c[k]
                    else:
                        new_dict[k] = c[k]
            return new_dict

        def _state_overlap_post_process(counts, num_qubits,
                                        num_shots, skip_idx=[]):
            """Post processes the results from state overlap circuit """

            # check that the number of qubits is even
            if not num_qubits % 2 == 0:
                raise AquaError("Input is not a valid shape.")

            # initialize variable to hold the state overlap estimate
            overlap = 0.0

            # Expectation value of controlled Z operation, see page 9 of Ref 38
            shift = num_qubits // 2

            keys = counts.keys()
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

            # do the state overlap (destructive swap test) post processing
            overlap = _state_overlap_post_process(pdip_count,
                                                  num_qubits,
                                                  num_shots,
                                                  skip_idx=[idx])

            all_zero_outcome = '0' * 2 * num_qubits
            prob = pdip_count[all_zero_outcome] / num_shots

            if not 0 <= prob <= 1:
                raise AquaError('Invalid probability')

            overlap *= prob
            scaled_overlap += overlap

        return initial_state_purity - scaled_overlap / num_working_qubits

    @staticmethod
    def _dip_circuit(num_qubits):
        """
        Construct DIP Test over 2n qubits
        Args:
            num_qubits (int): number of qubits in each of the states to be compared
        Returns:
            QuantumCircuit: the circuit
        """

        full_qr = QuantumRegister(2 * num_qubits)
        full_cr = ClassicalRegister(2 * num_qubits)

        qc = QuantumCircuit(full_qr, full_cr)

        # CNOTs
        for i in range(num_qubits):
            qc.cx(i + num_qubits, i)

        # Only meausure the first copy
        qc.measure(full_qr[:num_qubits], full_cr[:num_qubits])

        return qc

    @staticmethod
    def _swap_circuit(num_qubits):
        """
        Construct Destructive Swap Test over 2n qubits
        Args:
            num_qubits (int): number of qubits in each of the states to be compared
        Returns:
            QuantumCircuit: the circuit
        """

        full_qr = QuantumRegister(2 * num_qubits)
        full_cr = ClassicalRegister(2 * num_qubits)

        qc = QuantumCircuit(full_qr, full_cr)

        # CNOTs and Hadamards on second working qubit
        for i in range(num_qubits):
            qc.cx(i + num_qubits, i)
            qc.h(i + num_qubits)

        # Measure both copies
        qc.measure(full_qr, full_cr)

        return qc

    @staticmethod
    def _pdip_circuit(num_qubits, dip_qubit_idx):
        """Implements the partial dip test circuit over 2n qubits
        Args:
            num_qubits : int
                Number of quits
            dip_qubit_idx : list
                List of qubit indices (j in the paper) to do the pdip test on.

        Returns
            QuantumCircuit: the circuit
        """

        full_qr = QuantumRegister(2 * num_qubits)
        full_cr = ClassicalRegister(2 *num_qubits)

        pdip_qc = QuantumCircuit(full_qr, full_cr)

        swap_qubit_idx = list(set(range(num_qubits)) - set(dip_qubit_idx))
        swap_qubit_idx += [i + num_qubits for i in swap_qubit_idx]
        dip_qubit_idx += [i + num_qubits for i in dip_qubit_idx]

        nswap = len(swap_qubit_idx) // 2
        ndip = len(dip_qubit_idx) // 2

        if nswap > 0:
            swap_qc = VQSD._swap_circuit(nswap)
            pdip_qc.append(swap_qc, swap_qubit_idx, swap_qubit_idx)

        dip_qc = VQSD._dip_circuit(ndip)
        pdip_qc.append(dip_qc, dip_qubit_idx, dip_qubit_idx)

        return pdip_qc


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


if __name__ == '__main__':
    from qiskit.quantum_info.states import Statevector
    from qiskit.aqua.components.initial_states import Custom
    from qiskit.aqua.components.optimizers import COBYLA
    from qiskit import Aer

    state_vector = np.sqrt(1/2) * (Statevector.from_label('0+') +
                                   Statevector.from_label('1+'))

    initial_state = Custom(state_vector.num_qubits,
                           state_vector=state_vector.data)
    vqsd_obj = VQSD(initial_state, num_ancillae=0,
                    quantum_instance=Aer.get_backend("qasm_simulator"),
                    optimizer=COBYLA())

    print(vqsd_obj.print_settings())
    result = vqsd_obj.run(shots=8100)

    print(result)

