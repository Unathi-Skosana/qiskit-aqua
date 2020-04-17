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

    num_ancillae = 0
    state_vector = np.sqrt(7/10) * Statevector.from_label('++') \
        + np.sqrt(3/10) * Statevector.from_label('--')
    initial_state = Custom(state_vector.num_qubits,
                           state_vector=state_vector.data)
    vqsd_obj = VQSD(initial_state, q=1.0, num_ancillae=num_ancillae,
                    quantum_instance=Aer.get_backend("qasm_simulator"),
                    optimizer=COBYLA(), var_form=RY(initial_state._num_qubits,
                                                    depth=3))
    result = vqsd_obj.run(shots=8100)

    print("=== VQSD ===")
    print(result.eigenvalues)
    print(result.eigenstates)

    print("== Exact ===")
    density_mat = state_vector.to_operator()
    subsystem_density_mat = MatrixOperator(density_mat.data)
    exact_sys = NumPyEigensolver(subsystem_density_mat, k=4).run()
    print(exact_sys['eigenvalues'])
    print(exact_sys['eigenstates'])
