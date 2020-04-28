if __name__ == '__main__':
    import numpy as np
    from vqsd import VQSD
    from qiskit import Aer
    from qiskit.quantum_info.states import Statevector
    from qiskit.aqua.components.initial_states import Custom
    from qiskit.aqua.components.optimizers import COBYLA
    from qiskit.aqua.operators import MatrixOperator
    from qiskit.aqua.algorithms.eigen_solvers import NumPyEigensolver
    from qiskit.aqua.components.variational_forms import RY
    from qiskit.quantum_info import partial_trace

    num_ancillae = 1
    state_vector = np.sqrt(7/10) * Statevector.from_label('+1+') \
        + np.sqrt(3/10) * Statevector.from_label('-0-')
    initial_state = Custom(state_vector.num_qubits,
                           state_vector=state_vector.data)
    vqsd_obj = VQSD(initial_state, q=.5, num_ancillae=num_ancillae,
                    quantum_instance=Aer.get_backend("qasm_simulator"),
                    optimizer=COBYLA(), var_form=RY(initial_state._num_qubits -
                                                    num_ancillae,
                                                    depth=4))
    result = vqsd_obj.run(shots=5000)

    print("=== VQSD ===")
    print(result.eigenvalues)
    print(result.eigenstates)

    print("== Exact ===")
    density_mat = state_vector.to_operator()
    subsystem_density_mat = \
        partial_trace(MatrixOperator(density_mat.data).dense_matrix, [2])
    exact_sys = NumPyEigensolver(MatrixOperator(subsystem_density_mat.data), k=4).run()
    evals = exact_sys['eigenvalues']
    eigvecs = exact_sys['eigenstates']

    print(evals[0]*eigvecs[0],  eigvecs[0])
    print(evals[1]*eigvecs[1],  eigvecs[1])
    print(evals[2]*eigvecs[2],  eigvecs[2])
    print(evals[3]*eigvecs[3],  eigvecs[3])


