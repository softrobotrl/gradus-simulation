"""Checks for sofa_leg against MuJoCo on the real leg.xml.

Runs in the `sofa` conda env (SOFA's Python is 3.12). Skipped where SOFA or urdf/gradus/sim/leg.xml is missing.
The spec and servo tests only need mujoco + numpy; the scene tests build a SOFA leg.
"""

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LEG_XML = REPO / "urdf" / "gradus" / "sim" / "leg.xml"
HAVE_SOFA = importlib.util.find_spec("Sofa") is not None
HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAVE_MUJOCO and LEG_XML.exists(), "needs mujoco and urdf/gradus/sim/leg.xml")
class SpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco

        from sofa_leg.spec import leg_spec

        cls.mujoco = mujoco
        cls.spec = leg_spec(LEG_XML)
        cls.model = mujoco.MjModel.from_xml_path(str(LEG_XML))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def test_mass(self):
        self.assertAlmostEqual(self.spec.disk_mass, self.model.body_mass.sum(), places=12)

    def test_holes_are_mujoco_sites(self):
        for a, tendon in enumerate(self.spec.tendons):
            t = self.model.actuator_trnid[a, 0]
            sites = [self.model.wrap_objid[w] for w in range(self.model.tendon_adr[t],
                                                                 self.model.tendon_adr[t] + self.model.tendon_num[t])]
            np.testing.assert_allclose(tendon.world, self.data.site_xpos[sites], atol=1e-12)
            for disk, local, world in zip(tendon.disks, tendon.local, tendon.world):
                np.testing.assert_allclose(self.spec.disks[disk].to_world(local), world, atol=1e-12)

    def test_rest_lengths(self):
        for tendon in self.spec.tendons:
            length = np.linalg.norm(np.diff(tendon.world, axis=0), axis=1).sum()
            self.assertAlmostEqual(length, tendon.rest_length, places=12)

    def test_disks_cover_the_rod(self):
        start, end = self.spec.rod_span
        self.assertAlmostEqual(start, self.spec.disks[0].axial[0])
        self.assertAlmostEqual(end, self.spec.disks[-1].axial[1])
        for a, b in zip(self.spec.disks[:-1], self.spec.disks[1:]):
            self.assertLess(a.axial[1], b.axial[0], "disks must not overlap along the rod")

    def test_rod_stiffness_matches_leg_xml(self):
        """leg.xml's joint stiffness is E I / mean pitch with the same rod; recover it from the spec's defaults."""
        centres = [self.spec.axial_coordinate(d.pos) for d in self.spec.disks[1:]]
        anchors = np.array([self.data.xanchor[j] for j in range(self.model.njnt)])
        pitch = np.diff(self.spec.axial_coordinate(anchors)).mean()
        self.assertGreater(len(centres), 1)
        # within 2%: Daniel's merge.py averages the pitch slightly differently from these joint anchors
        self.assertAlmostEqual(self.spec.rod.bending_stiffness / pitch / self.spec.joint_stiffness, 1.0, delta=0.02)


class RodMeshTest(unittest.TestCase):
    def test_section_matches_circle(self):
        from sofa_leg.rodmesh import section_properties

        for n in (2, 3, 4):
            props = section_properties(0.004, n)
            self.assertAlmostEqual(props["I"], 1.0, places=12)
            self.assertAlmostEqual(props["area"] * props["density_scale"], 1.0, places=12)

    def test_layers_keep_breakpoints(self):
        from sofa_leg.rodmesh import axial_layers

        layers = axial_layers([0.0, 0.013, 0.1], 0.01)
        self.assertTrue(np.isclose(layers, 0.013).any())
        self.assertLessEqual(np.diff(layers).max(), 0.01 + 1e-12)

    def test_hexahedra_are_positive(self):
        from sofa_leg.rodmesh import axial_layers, rod_mesh

        nodes, hexes, _ = rod_mesh(0.004, axial_layers([0, 0.02], 0.005), 3, axis=(0, 0, -1))
        for h in hexes:
            p = nodes[h]
            volume = np.dot(np.cross(p[1] - p[0], p[3] - p[0]), p[4] - p[0])
            self.assertNotEqual(np.sign(volume), 0)
        signs = {np.sign(np.dot(np.cross(nodes[h[1]] - nodes[h[0]], nodes[h[3]] - nodes[h[0]]),
                                nodes[h[4]] - nodes[h[0]])) for h in hexes}
        self.assertEqual(len(signs), 1, "every hexahedron must have the same orientation")


@unittest.skipUnless(HAVE_MUJOCO and LEG_XML.exists(), "needs mujoco and urdf/gradus/sim/leg.xml")
class ServoTest(unittest.TestCase):
    def test_servo_law_matches_mujoco(self):
        import mujoco

        from sofa_leg.spec import leg_spec

        if not HAVE_SOFA:
            self.skipTest("sofa_leg.scene imports Sofa")
        from sofa_leg.scene import servo_tension

        model = mujoco.MjModel.from_xml_path(str(LEG_XML))
        data = mujoco.MjData(model)
        spec = leg_spec(LEG_XML)
        rng = np.random.default_rng(0)
        kp, kv, fmax = (np.array([getattr(t, k) for t in spec.tendons]) for k in ("kp", "kv", "fmax"))
        for _ in range(20):
            data.qpos[:] = model.qpos0
            for j in range(model.njnt):
                axis = rng.normal(size=3)
                q = np.zeros(4)
                mujoco.mju_axisAngle2Quat(q, axis / np.linalg.norm(axis), rng.uniform(0, 0.05))
                data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 4] = q
            data.qvel[:] = rng.normal(scale=0.5, size=model.nv)
            data.ctrl[:] = rng.uniform(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_forward(model, data)
            ours = servo_tension(data.ten_length, data.ten_velocity, data.ctrl, kp, kv, fmax)
            np.testing.assert_allclose(ours, -data.actuator_force, atol=1e-9)

    def test_quat_rotate_matches_mujoco(self):
        import mujoco

        if not HAVE_SOFA:
            self.skipTest("sofa_leg.scene imports Sofa")
        from sofa_leg.scene import quat_rotate

        rng = np.random.default_rng(1)
        for _ in range(10):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            v = rng.normal(size=3)
            expected = np.zeros(3)
            mujoco.mju_rotVecQuat(expected, v, q)
            np.testing.assert_allclose(quat_rotate(np.r_[q[1:], q[0]], v), expected, atol=1e-12)


@unittest.skipUnless(HAVE_SOFA and HAVE_MUJOCO and LEG_XML.exists(), "needs SOFA, mujoco and leg.xml")
class SofaLegTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sofa_leg.leg import SofaLeg

        cls.leg = SofaLeg(LEG_XML)

    def test_rest_lengths_match_mujoco(self):
        self.leg.reset()
        expected = [t.rest_length for t in self.leg.spec.tendons]
        np.testing.assert_allclose(self.leg.length(), expected, atol=1e-9)
        np.testing.assert_allclose(self.leg.tension(), 0.0, atol=1e-9)

    def test_steps_stay_finite_and_pull_moves_the_foot(self):
        self.leg.reset()
        start = self.leg.tip()
        self.leg.set_pull([0.5, 0, 0, 0, 0, 0, 0, 0])
        self.leg.step(50)
        self.assertTrue(self.leg.finite())
        self.assertGreater(np.linalg.norm(self.leg.tip() - start), 1e-4)
        self.assertGreater(self.leg.tension()[0], 0)

    def test_length_rate_is_the_derivative(self):
        self.leg.reset()
        self.leg.set_pull([0, 0.6, 0, 0, 0, 0.3, 0, 0])
        self.leg.step(20)
        before, rate = self.leg.length(), self.leg.length_rate()
        self.leg.step(1)
        after, rate_after = self.leg.length(), self.leg.length_rate()
        numeric = (after - before) / self.leg.dt
        np.testing.assert_allclose(numeric, 0.5 * (rate + rate_after), atol=2e-3 + 0.05 * np.abs(numeric).max())

    def test_restore_is_repeatable(self):
        self.leg.reset()
        self.leg.set_pull([0.2, 0, 0.1, 0, 0, 0, 0, 0])
        self.leg.step(10)
        state = self.leg.get_state()

        def run():
            self.leg.set_state(state)
            self.leg.set_pull([0, 0.5, 0, 0, 0, 0, 0.3, 0])
            self.leg.step(15)
            return self.leg.tip()

        first = run()
        self.leg.set_pull([0.9, 0, 0, 0.9, 0, 0, 0, 0])
        self.leg.step(10)   # unrelated history in between
        np.testing.assert_allclose(run(), first, atol=1e-6)

    def test_every_disk_holds_rod_nodes(self):
        owner = self.leg.handle.owner
        self.assertEqual(set(owner[owner >= 0]), set(range(len(self.leg.spec.disks))))


if __name__ == "__main__":
    unittest.main()
