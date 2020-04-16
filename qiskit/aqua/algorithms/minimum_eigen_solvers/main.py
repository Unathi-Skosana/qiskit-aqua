if __name__ == '__main__':
    import numpy as np
    from vqsd import VQSD
    from qiskit import Aer
    from qiskit.quantum_info.states import Statevector
    from qiskit.aqua.components.initial_states import Custom
    from qiskit.aqua.components.optimizers import COBYLA
    from qiskit.aqua.operators import MatrixOperator
    from qiskit.aqua.algorithms.eigen_solvers import NumPyEigensolver
    from qiskit.quantum_info import partial_trace

    num_ancillae = 1
    state_vector = np.sqrt(7/10) * Statevector.from_label('0+') \
        + np.sqrt(3/10) * Statevector.from_label('1-')
    initial_state = Custom(state_vector.num_qubits,
                           state_vector=state_vector.data)
    vqsd_obj = VQSD(initial_state, num_ancillae=num_ancillae,
                    quantum_instance=Aer.get_backend("qasm_simulator"),
                    optimizer=COBYLA())
    result = vqsd_obj.run(shots=8100)

    print("=== VQSD ===")
    print(result.eigenvalue)
    print(result.eigenstate)

    print("== Exact ===")
    density_mat = state_vector.to_operator()
    subsystem_density_mat = MatrixOperator(partial_trace(density_mat.data,
                                                         [1]).data)
    exact_sys = NumPyEigensolver(subsystem_density_mat, k=2).run()
    print(exact_sys['eigenvalues'])
    print(exact_sys['eigenstates'])
