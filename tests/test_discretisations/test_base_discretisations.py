#
# Tests for the base model class
#
import pybamm

import numpy as np
import unittest


class MeshForTesting(object):
    def __init__(self):
        self.whole_cell = SubMeshForTesting(np.linspace(0, 1, 100))
        self.negative_electrode = SubMeshForTesting(self.whole_cell.centres[:40])


class SubMeshForTesting(object):
    def __init__(self, centres):
        self.centres = centres
        self.npts = centres.size


class DiscretisationForTesting(pybamm.MatrixVectorDiscretisation):
    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient_matrix(self, domain):
        n = getattr(self.mesh, domain[0]).npts
        return pybamm.Matrix(np.eye(n))

    def divergence_matrix(self, domain):
        n = getattr(self.mesh, domain[0]).npts
        return pybamm.Matrix(np.eye(n))


class ModelForTesting(object):
    def __init__(self, rhs, initial_conditions, boundary_conditions):
        self.rhs = rhs
        self.initial_conditions = initial_conditions
        self.boundary_conditions = boundary_conditions


class TestDiscretise(unittest.TestCase):
    def test_discretise_slicing(self):
        # One variable
        mesh = MeshForTesting()
        disc = pybamm.BaseDiscretisation(mesh)
        c = pybamm.Variable("c", domain=["whole_cell"])
        variables = [c]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(y_slices, {c.id: slice(0, 100)})
        c_true = mesh.whole_cell.centres ** 2
        y = c_true
        np.testing.assert_array_equal(y[y_slices[c.id]], c_true)

        # Several variables
        d = pybamm.Variable("d", domain=["whole_cell"])
        jn = pybamm.Variable("jn", domain=["negative_electrode"])
        variables = [c, d, jn]
        y_slices = disc.get_variable_slices(variables)
        self.assertEqual(
            y_slices,
            {c.id: slice(0, 100), d.id: slice(100, 200), jn.id: slice(200, 240)},
        )
        d_true = 4 * mesh.whole_cell.centres
        jn_true = mesh.negative_electrode.centres ** 3
        y = np.concatenate([c_true, d_true, jn_true])
        np.testing.assert_array_equal(y[y_slices[c.id]], c_true)
        np.testing.assert_array_equal(y[y_slices[d.id]], d_true)
        np.testing.assert_array_equal(y[y_slices[jn.id]], jn_true)

    def test_process_symbol_base(self):
        disc = pybamm.BaseDiscretisation(None)

        # variable
        var = pybamm.Variable("var")
        y_slices = {var.id: slice(53)}
        var_disc = disc.process_symbol(var, None, y_slices, None)
        self.assertTrue(isinstance(var_disc, pybamm.VariableVector))
        self.assertEqual(var_disc._y_slice, y_slices[var.id])
        # scalar
        scal = pybamm.Scalar(5)
        scal_disc = disc.process_symbol(scal, None, None, None)
        self.assertTrue(isinstance(scal_disc, pybamm.Scalar))
        self.assertEqual(scal_disc.value, scal.value)

        # parameter
        par = pybamm.Parameter("par")
        with self.assertRaises(TypeError):
            disc.process_symbol(par, None, None, None)

        # binary operator
        bin = pybamm.BinaryOperator("bin", var, scal)
        bin_disc = disc.process_symbol(bin, None, y_slices, None)
        self.assertTrue(isinstance(bin_disc, pybamm.BinaryOperator))
        self.assertTrue(isinstance(bin_disc.left, pybamm.VariableVector))
        self.assertTrue(isinstance(bin_disc.right, pybamm.Scalar))

        # non-spatial unary operator
        un = pybamm.UnaryOperator("un", var)
        un_disc = disc.process_symbol(un, None, y_slices, None)
        self.assertTrue(isinstance(un_disc, pybamm.UnaryOperator))
        self.assertTrue(isinstance(un_disc.child, pybamm.VariableVector))

    def test_discretise_spatial_operator(self):
        # no boundary conditions
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)
        var = pybamm.Variable("var", domain=["whole_cell"])
        y_slices = disc.get_variable_slices([var])
        for eqn in [pybamm.grad(var), pybamm.div(var)]:
            eqn_disc = disc.process_symbol(eqn, var.domain, y_slices, {})

            self.assertTrue(isinstance(eqn_disc, pybamm.MatrixMultiplication))
            self.assertTrue(isinstance(eqn_disc.left, pybamm.Matrix))
            self.assertTrue(isinstance(eqn_disc.right, pybamm.VariableVector))

            y = mesh.whole_cell.centres ** 2
            var_disc = disc.process_symbol(var, None, y_slices, None)
            # grad and var are identity operators here (for testing purposes)
            np.testing.assert_array_equal(eqn_disc.evaluate(y), var_disc.evaluate(y))

        # with boundary conditions

    def test_process_initial_conditions(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole_cell"])
        initial_conditions = {c: pybamm.Scalar(3)}
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)
        y0 = disc.process_initial_conditions(initial_conditions)
        np.testing.assert_array_equal(y0, 3 * np.ones_like(mesh.whole_cell.centres))

        # two equations
        T = pybamm.Variable("T", domain=["negative_electrode"])
        initial_conditions = {c: pybamm.Scalar(3), T: pybamm.Scalar(5)}
        y0 = disc.process_initial_conditions(initial_conditions)
        np.testing.assert_array_equal(
            y0,
            np.concatenate(
                [
                    3 * np.ones_like(mesh.whole_cell.centres),
                    5 * np.ones_like(mesh.negative_electrode.centres),
                ]
            ),
        )

    def test_process_rhs(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole_cell"])
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        boundary_conditions = {N: (pybamm.Scalar(0), pybamm.Scalar(2))}
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)

        y = mesh.whole_cell.centres ** 2
        y_slices = disc.get_variable_slices(rhs.keys())
        dydt = disc.process_rhs(rhs, boundary_conditions, y_slices)
        np.testing.assert_array_equal(y[1:-1], dydt(y)[1:-1])
        self.assertEqual(dydt(y)[0], 0)
        self.assertEqual(dydt(y)[-1], 2)

        # two equations
        # T = pybamm.Variable("T", domain=["negative_electrode"])
        # q = pybamm.grad(T)
        # rhs = {c: pybamm.div(N), T: pybamm.div(q)}
        # boundary_conditions = {
        #     N: (pybamm.Scalar(0), pybamm.Scalar(2)),
        #     q: (pybamm.Scalar(-3), pybamm.Scalar(12)),
        # }
        # model = ModelForTesting(rhs, initial_conditions, boundary_conditions)

        # y0, dydt = disc.process_model(model)

    @unittest.skip("")
    def test_process_model(self):
        # one equation
        c = pybamm.Variable("c", domain=["whole_cell"])
        N = pybamm.grad(c)
        rhs = {c: pybamm.div(N)}
        initial_conditions = {c: pybamm.Scalar(1)}
        boundary_conditions = {N: (pybamm.Scalar(0), pybamm.Scalar(2))}
        model = ModelForTesting(rhs, initial_conditions, boundary_conditions)
        mesh = MeshForTesting()
        disc = DiscretisationForTesting(mesh)

        y0, dydt = disc.process_model(model)
        np.testing.assert_array_equal(y0, np.ones_like(mesh.whole_cell.centres))
        np.testing.assert_array_equal(y0[1:-1], dydt(y0)[1:-1])
        np.testing.assert_array_equal(dydt(y0)[0], 0)
        np.testing.assert_array_equal(dydt(y0)[-1], 2)

        # two equations
        T = pybamm.Variable("T", domain=["negative_electrode"])
        q = pybamm.grad(T)
        rhs[T] = pybamm.div(q)
        initial_conditions[T] = pybamm.Scalar(5)
        boundary_conditions[q] = (pybamm.Scalar(-3), pybamm.Scalar(12))
        model = ModelForTesting(rhs, initial_conditions, boundary_conditions)

        # y0, dydt = disc.process_model(model)

    @unittest.skip("")
    def test_concatenation(self):
        pass


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
