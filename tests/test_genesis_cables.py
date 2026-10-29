"""Cable mechanics checks; these do not require initializing Genesis."""

import unittest

import torch

from genesis_leg.cables import cable_forces, rotate


class CableTests(unittest.TestCase):
    def test_slack_cable_does_not_push(self):
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
        forces, tension = cable_forces(points, points * 0, 2., 100., 1.)
        self.assertEqual(tension.item(), 0)
        self.assertTrue(torch.equal(forces, torch.zeros_like(forces)))

    def test_guide_forces_conserve_force_and_torque(self):
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]])
        forces, tension = cable_forces(points, points * 0, 1.9, 100., 1.)
        self.assertAlmostEqual(tension.item(), 10., places=4)
        torch.testing.assert_close(forces.sum(-2), torch.zeros(1, 3))
        torch.testing.assert_close(torch.linalg.cross(points, forces).sum(-2), torch.zeros(1, 3))
        torch.testing.assert_close(forces[0, 1], torch.tensor([-10., 10., 0.]))

    def test_forces_are_negative_gradient_of_elastic_energy(self):
        points = torch.tensor([[[0., 0., 0.], [.8, .1, 0.], [1., 1., .2]]], requires_grad=True)
        length = torch.linalg.vector_norm(points[:, 1:] - points[:, :-1], dim=-1).sum(-1)
        energy = .5 * 50 * (length - 1.).square().sum()
        (gradient,) = torch.autograd.grad(energy, points)
        force, _ = cable_forces(points.detach(), points.detach() * 0, 1., 50., 0.)
        torch.testing.assert_close(force, -gradient)

    def test_damping_opposes_extension_rate(self):
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
        velocity = torch.tensor([[[0., 0., 0.], [.2, 0., 0.]]])
        force, tension = cable_forces(points, velocity, .9, 100., 5.)
        self.assertAlmostEqual(tension.item(), 11., places=4)
        self.assertLess((force * velocity).sum().item(), 0)

    def test_wxyz_rotation(self):
        quat = torch.tensor([2**-.5, 0., 0., 2**-.5])
        torch.testing.assert_close(rotate(quat, torch.tensor([1., 0., 0.])), torch.tensor([0., 1., 0.]), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
