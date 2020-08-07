if __name__ == '__main__':
    import numpy as np
    from qiskit import Aer
    from qiskit.aqua.algorithms.minimum_eigen_solvers import VQSD
    from qiskit.quantum_info.states import Statevector
    from qiskit.aqua.components.initial_states import Custom
    from qiskit.aqua.components.optimizers import COBYLA
    from qiskit.aqua.operators import MatrixOperator
    from qiskit.aqua.algorithms.eigen_solvers import NumPyEigensolver
    from qiskit.aqua.components.variational_forms import RY
    from qiskit.quantum_info import partial_trace

    num_ancillae = 1
    state_vector = np.sqrt(6/10) * Statevector.from_label('+1+') \
        + np.sqrt(4/10) * Statevector.from_label('-0-')
    initial_state = Custom(state_vector.num_qubits,
                           state_vector=state_vector.data)
    vqsd_obj = VQSD(initial_state, q=.25, num_ancillae=num_ancillae,
                    quantum_instance=Aer.get_backend("qasm_simulator"),
                    optimizer=COBYLA(), var_form=RY(initial_state._num_qubits -
                                                    num_ancillae,
                                                    depth=2))
    result = vqsd_obj.run(shots=1000)

    print("=== VQSD ===")
    print(result.eigenvalue)
    print(result.eigenstate)

    print("== Exact ===")
    density_mat = state_vector.to_operator()
    subsystem_density_mat = \
        partial_trace(MatrixOperator(density_mat.data).dense_matrix, [2])
    exact_sys = NumPyEigensolver(MatrixOperator(subsystem_density_mat.data), k=4).run()
    eigvals = exact_sys['eigenvalues']
    eigvecs = exact_sys['eigenstates']

    print(eigvals)
    print(eigvecs)

    ss = 0
    for i in range(0, 4):
        projector = np.eye(4) - \
        np.outer(np.transpose(np.conj(result.eigenstate[i])),
                result.eigenstate[i])
        s = projector @ subsystem_density_mat.data @ result.eigenstate[i]
        ss += np.inner(s,s)
    print("=== Error ===")
    print(ss)
