"""Merged leg.xml -> Genesis routes and servo forces, checked against MuJoCo. These do not initialize Genesis."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

from genesis_leg.leg_fixture import OUTER_STATIONS, N_STATIONS, write_fixture
from genesis_leg.leg_model import find_tip, prepare_leg, read_routes
from genesis_leg.leg_tendons import pad_routes, tendon_forces


class LegCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.xml = write_fixture(Path(cls.directory.name) / "leg")
        cls.model = mujoco.MjModel.from_xml_path(str(cls.xml))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()


class PrepareTests(LegCase):
    def test_routes_match_compiled_tendons(self):
        _, routes = prepare_leg(self.xml)
        tendons = routes["tendons"]
        self.assertEqual([len(t["bodies"]) for t in tendons], [N_STATIONS] * 4 + [OUTER_STATIONS] * 4)
        self.assertEqual(tendons[0]["name"], "tendon_inner_front")
        self.assertEqual(tendons[0]["bodies"][0], "hip")
        self.assertEqual(tendons[0]["bodies"][-1], "foot")
        self.assertEqual((tendons[0]["kp"], tendons[0]["kv"], tendons[0]["fmax"]), (5000.0, 50.0, 200.0))
        self.assertAlmostEqual(tendons[0]["offsets"][0][0], 0.016)

    def test_genesis_model_keeps_bodies_and_mass_but_not_tendons(self):
        out, routes = prepare_leg(self.xml)
        root = ET.parse(out).getroot()
        self.assertIsNone(root.find("tendon"))
        self.assertIsNone(root.find("actuator"))
        self.assertTrue(all(g.get("contype") == "0" for g in root.iter("geom")))
        stripped = mujoco.MjModel.from_xml_path(str(out))
        self.assertEqual((stripped.nbody, stripped.njnt, stripped.nu), (self.model.nbody, self.model.njnt, 0))
        self.assertAlmostEqual(stripped.body_mass.sum(), self.model.body_mass.sum(), places=9)
        self.assertAlmostEqual(routes["mass"], 1.24, places=3)
        self.assertEqual(json.loads(out.with_name("leg_routes.json").read_text())["tip_body"], "foot")

    def test_tip_defaults_to_centre_of_last_holes(self):
        body, local = find_tip(self.model, read_routes(self.model))
        self.assertEqual(body, "foot")
        np.testing.assert_allclose(local, [0, 0, 0], atol=1e-12)

    def test_force_motors_are_rejected(self):
        root = ET.fromstring(self.xml.read_text())
        actuator = root.find("actuator")
        for element in list(actuator):
            ET.SubElement(actuator, "motor", name=f"m_{element.get('name')}", tendon=element.get("tendon"), gear="-1")
            actuator.remove(element)
        path = self.xml.with_name("motors.xml")
        path.write_text(ET.tostring(root, encoding="unicode"))
        with self.assertRaisesRegex(ValueError, "--motors"):
            prepare_leg(path)


class ForceTests(LegCase):
    def state(self, seed):
        """A random bent, moving pose in MuJoCo, plus its tendons' routed points and velocities as tensors."""
        rng = np.random.default_rng(seed)
        model = self.model
        data = mujoco.MjData(model)
        for j in range(model.njnt):
            axis = rng.normal(size=3)
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, axis / np.linalg.norm(axis), rng.uniform(0, 0.1))
            data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 4] = quat
        data.qvel[:] = rng.normal(size=model.nv) * 0.2
        low, high = model.actuator_ctrlrange.T
        data.ctrl[:] = low + rng.uniform(size=model.nu) * (high - low)
        mujoco.mj_forward(model, data)

        routes, velocities, sites = [], [], []
        velocity = np.zeros(6)
        for tendon in read_routes(model):
            ids = [model.site(f"{tendon['name']}_{i:02d}").id for i in range(len(tendon["bodies"]))]
            sites.append(ids)
            routes.append(data.site_xpos[ids].copy())
            row = []
            for site in ids:
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0)
                row.append(velocity[3:].copy())
            velocities.append(np.array(row))
        points, mask = pad_routes(routes)
        speeds, _ = pad_routes(velocities)
        return data, points, speeds, mask, sites

    def forces(self, data, points, speeds, mask):
        tendons = read_routes(self.model)
        kp, kv, fmax = (torch.tensor([t[name] for t in tendons], dtype=torch.float64) for name in ("kp", "kv", "fmax"))
        return tendon_forces(points, speeds, mask, torch.as_tensor(data.ctrl), kp, kv, fmax)

    def test_length_rate_and_tension_match_mujoco(self):
        for seed in range(5):
            data, points, speeds, mask, _ = self.state(seed)
            _, tension, length, rate = self.forces(data, points, speeds, mask)
            np.testing.assert_allclose(length.numpy(), data.ten_length, atol=1e-9)
            np.testing.assert_allclose(rate.numpy(), data.ten_velocity, atol=1e-9)
            np.testing.assert_allclose(tension.numpy(), -data.actuator_force, atol=1e-6)
            self.assertTrue((tension > 0).any())

    def test_node_forces_give_mujocos_joint_torques(self):
        for seed in range(5):
            data, points, speeds, mask, sites = self.state(seed)
            force, *_ = self.forces(data, points, speeds, mask)
            torque = np.zeros(self.model.nv)
            jacobian = np.zeros((3, self.model.nv))
            for t, ids in enumerate(sites):
                for n, site in enumerate(ids):
                    mujoco.mj_jacSite(self.model, data, jacobian, None, site)
                    torque += jacobian.T @ force[t, n].numpy()
            np.testing.assert_allclose(torque, data.qfrc_actuator, atol=1e-6)

    def test_slack_tendon_does_not_push(self):
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]])
        mask = torch.tensor([[True, True, False]])
        forces, tension, length, _ = tendon_forces(
            points, points * 0, mask, torch.tensor([2.]), torch.tensor([100.]), torch.tensor([1.]), torch.tensor([50.]),
        )
        self.assertEqual(tension.item(), 0)
        self.assertEqual(length.item(), 1)
        self.assertTrue(torch.equal(forces, torch.zeros_like(forces)))

    def test_padding_adds_no_force_and_tension_saturates(self):
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]])
        mask = torch.tensor([[True, True, False]])
        forces, tension, *_ = tendon_forces(
            points, points * 0, mask, torch.tensor([.5]), torch.tensor([100.]), torch.tensor([1.]), torch.tensor([20.]),
        )
        self.assertEqual(tension.item(), 20)
        torch.testing.assert_close(forces[0, 0], torch.tensor([20., 0., 0.]))
        torch.testing.assert_close(forces[0, 1], torch.tensor([-20., 0., 0.]))
        self.assertTrue(torch.equal(forces[0, 2], torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()
