import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    FieldStatistics,
    batches,
    build_latent_windows,
    decoder_gradient_is_clear,
    normalize_coords,
    octahedral_transform,
)


class NormalizationTests(unittest.TestCase):
    def test_field_normalization_matches_upstream_dim_zero_semantics(self):
        data = torch.tensor(
            [
                [[0.0, 2.0], [10.0, -4.0], [3.0, 8.0]],
                [[2.0, 6.0], [14.0, 0.0], [7.0, 12.0]],
                [[1.0, 4.0], [12.0, -2.0], [5.0, 10.0]],
            ]
        )
        minimum = data.amin(dim=0)
        maximum = data.amax(dim=0)
        stats = FieldStatistics(
            minimum=minimum,
            maximum=maximum,
            benchmark_mean=data.mean(dim=(0, 1)),
            benchmark_std=data.std(dim=(0, 1)),
        )
        expected = (data - minimum) / (maximum - minimum) * 2.0 - 1.0
        self.assertTrue(torch.equal(stats.normalize(data), expected))
        self.assertTrue(torch.equal(stats.denormalize(expected), data))

        indices = torch.tensor([2, 0])
        selected = data[1, indices]
        expected_selected = expected[1, indices]
        self.assertTrue(
            torch.equal(stats.normalize(selected, indices), expected_selected)
        )

        upstream_root = Path(
            "/projects/ammoniacomb/generative_reconstruction/baselines/CoNFiLD"
        )
        sys.path.insert(0, str(upstream_root))
        from ConditionalNeuralField.cnf.utils.normalize import Normalizer_ts

        upstream = Normalizer_ts(method="-11", dim=0)
        upstream_normalized = upstream.fit_normalize(data)
        self.assertTrue(torch.equal(upstream_normalized, expected))

    def test_coordinate_normalization_is_minus_one_to_one(self):
        coords = torch.tensor([[2.0, 10.0, -3.0], [4.0, 14.0, 1.0]])
        normalized, lower, upper = normalize_coords(coords)
        self.assertTrue(torch.equal(normalized[0], -torch.ones(3)))
        self.assertTrue(torch.equal(normalized[1], torch.ones(3)))
        self.assertTrue(torch.equal(lower, coords[0]))
        self.assertTrue(torch.equal(upper, coords[1]))


class SymmetryTests(unittest.TestCase):
    def setUp(self):
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(3.0),
                torch.arange(3.0),
                torch.arange(3.0),
                indexing="ij",
            ),
            dim=-1,
        )
        scalar = (
            100.0 * grid[..., 0] + 10.0 * grid[..., 1] + grid[..., 2]
        ).unsqueeze(-1)
        velocity = torch.stack(
            [grid[..., 0] + 1.0, 3.0 * grid[..., 1] + 2.0, 7.0 * grid[..., 2] + 4.0],
            dim=-1,
        )
        self.fields = torch.cat([velocity, scalar], dim=-1).reshape(-1, 4)

    def test_all_48_transforms_are_unique(self):
        outputs = {
            octahedral_transform(self.fields, (3, 3, 3), group).numpy().tobytes()
            for group in range(48)
        }
        self.assertEqual(len(outputs), 48)

    def test_every_transform_has_an_inverse(self):
        for group in range(48):
            transformed = octahedral_transform(self.fields, (3, 3, 3), group)
            has_inverse = any(
                torch.equal(
                    octahedral_transform(transformed, (3, 3, 3), candidate),
                    self.fields,
                )
                for candidate in range(48)
            )
            self.assertTrue(has_inverse, f"No inverse found for group element {group}")


class OptimizerAndEvaluationTests(unittest.TestCase):
    def test_latent_windows_do_not_cross_cube_or_group_boundaries(self):
        latents = torch.arange(12.0).reshape(6, 2)
        windows, manifest = build_latent_windows(
            latents,
            n_snapshots=3,
            n_groups=2,
            cube_length=3,
            window_length=2,
        )
        self.assertEqual(tuple(windows.shape), (4, 1, 2, 2))
        expected = torch.stack([latents[0], latents[2]])
        self.assertTrue(torch.equal(windows[0, 0], expected))
        self.assertEqual(manifest[0]["group"], 0)
        self.assertEqual(manifest[2]["group"], 1)

    def test_balanced_batches_visit_every_item_once(self):
        permutation = torch.randperm(101)
        visited = torch.cat(list(batches(permutation, 8)))
        self.assertTrue(torch.equal(visited, permutation))
        self.assertEqual(torch.unique(visited).numel(), 101)

    def test_frozen_decoder_does_not_accumulate_gradients(self):
        decoder = torch.nn.Sequential(
            torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
        )
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        latent = torch.nn.Parameter(torch.zeros(1, 5))
        target = torch.ones(1, 2)
        optimizer = torch.optim.Adam([latent], lr=1e-3)
        for _ in range(3):
            loss = torch.nn.functional.mse_loss(decoder(latent), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertTrue(decoder_gradient_is_clear(decoder))
        self.assertIsNotNone(latent.grad)

    def test_two_timescale_parameter_delta_matches_upstream_loop(self):
        torch.manual_seed(5)
        model_a = torch.nn.Linear(2, 1, bias=False)
        model_b = torch.nn.Linear(2, 1, bias=False)
        model_b.load_state_dict(model_a.state_dict())
        latent_a = torch.nn.Parameter(torch.zeros(2, 2))
        latent_b = torch.nn.Parameter(torch.zeros(2, 2))
        net_a = torch.optim.Adam(model_a.parameters(), lr=1e-4)
        net_b = torch.optim.Adam(model_b.parameters(), lr=1e-4)
        lat_a = torch.optim.Adam([latent_a], lr=1e-5)
        lat_b = torch.optim.Adam([latent_b], lr=1e-5)
        targets = [torch.tensor([[1.0]]), torch.tensor([[-1.0]])]

        # End-of-epoch form used by the adapter.
        net_a.zero_grad(set_to_none=True)
        for index, target in enumerate(targets):
            loss = torch.nn.functional.mse_loss(model_a(latent_a[index:index + 1]), target)
            lat_a.zero_grad(set_to_none=True)
            loss.backward()
            lat_a.step()
        net_a.step()

        # Upstream's next-epoch-boundary step applied to the same accumulated grads.
        net_b.zero_grad(set_to_none=True)
        for index, target in enumerate(targets):
            loss = torch.nn.functional.mse_loss(model_b(latent_b[index:index + 1]), target)
            lat_b.zero_grad(set_to_none=True)
            loss.backward()
            lat_b.step()
        net_b.step()

        self.assertTrue(torch.equal(model_a.weight, model_b.weight))
        self.assertTrue(torch.equal(latent_a, latent_b))


if __name__ == "__main__":
    unittest.main()
