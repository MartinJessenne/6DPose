import dataclasses
import typing
import unittest

import optuna

import cli_config
from methods.base import UnknownOverrideError, _unwrap_optional
from methods.ppf import PPFParams
from methods.ransac import RansacParams
from methods.ransac3dof import Ransac3DoFParams
from methods.vsac_se2 import VSACSe2Params


class TestParamsAuthority(unittest.TestCase):
    def test_t1_every_field_survives(self):
        ransac_fields = {f.name for f in dataclasses.fields(RansacParams)}
        expected_ransac = {
            "seed",
            "voxel_size",
            "ransac_max_iterations",
            "icp_max_correspondence_distance",
            "icp_max_iterations",
        }
        self.assertEqual(ransac_fields, expected_ransac)

        r3dof_fields = {f.name for f in dataclasses.fields(Ransac3DoFParams)}
        expected_r3dof = expected_ransac | {
            "z_offset",
            "z_gate_threshold",
            "edge_length_tolerance",
            "ransac_confidence",
            "front_crop_aspect",
            "front_face_max_angle_deg",
            "hoppe_normal_orientation",
            "icp_visibility_cull",
            "icp_gnc_scale_min",
            "icp_gnc_shrink",
        }
        self.assertEqual(r3dof_fields, expected_r3dof)

    def test_t2_search_space_declared(self):
        self.assertEqual(
            set(RansacParams.search_space().keys()),
            {"icp_max_correspondence_distance"},
        )
        self.assertEqual(
            set(PPFParams.search_space().keys()),
            {
                "ppf_sampling_step",
                "ppf_distance_step",
                "ppf_match_threshold",
                "icp_max_correspondence_distance",
            },
        )
        self.assertEqual(
            set(Ransac3DoFParams.search_space().keys()),
            {
                "icp_max_correspondence_distance",
                "z_gate_threshold",
            },
        )
        self.assertEqual(
            set(VSACSe2Params.search_space().keys()),
            {
                "icp_max_correspondence_distance",
                "z_gate_threshold",
                "rho",
            },
        )

    def test_t3_sweep_starts_from_profile(self):
        profile = RansacParams(voxel_size=0.03)
        trial = optuna.trial.FixedTrial(
            {"ransac_max_iterations": 5000, "icp_max_correspondence_distance": 0.05}
        )
        sampled = RansacParams.sample_optuna(trial, base=profile, fixed={"voxel_size"})
        self.assertEqual(sampled.voxel_size, 0.03)
        self.assertEqual(sampled.ransac_max_iterations, 100000)

    def test_t4_no_profile_outside_its_own_range(self):
        """Every tuned profile must construct. __post_init__ rejects a value outside
        its own declared SearchRange, and a tuned optimum sitting outside the range
        being searched means the sweep can never rediscover it."""
        selects = [n for n in dir(cli_config) if n.endswith("ProfileSelect")]
        self.assertGreaterEqual(len(selects), 4)
        checked = 0
        for select_name in selects:
            for member in typing.get_args(getattr(cli_config, select_name)):
                profile_cls, subcommand = typing.get_args(member)
                default = subcommand.default
                profile = default if isinstance(default, profile_cls) else profile_cls()
                with self.subTest(select=select_name, profile=subcommand.name):
                    type(profile.params)(**dataclasses.asdict(profile.params))
                checked += 1
        self.assertGreater(checked, 0)

    def test_t5_override_round_trip_by_type(self):
        p = Ransac3DoFParams().with_overrides(
            icp_visibility_cull="true",
            front_face_max_angle_deg="60.0",
            z_offset="None",
            front_crop_aspect="2.5",
            icp_gnc_scale_min="0.0148",
        )
        self.assertIs(p.icp_visibility_cull, True)
        self.assertEqual(p.front_face_max_angle_deg, 60.0)
        self.assertIsNone(p.z_offset)
        self.assertEqual(p.front_crop_aspect, 2.5)
        self.assertEqual(p.icp_gnc_scale_min, 0.0148)

    def test_t6_unknown_name_raises(self):
        with self.assertRaises(UnknownOverrideError) as ctx:
            RansacParams().with_overrides(unknown_field="123")
        self.assertEqual(ctx.exception.name, "unknown_field")

    def test_t7_what_ran_is_what_was_recorded(self):
        study = optuna.create_study()
        trial = study.ask()
        params = RansacParams.sample_optuna(trial)
        for name in RansacParams.search_space():
            self.assertIn(name, trial.params)
            self.assertEqual(getattr(params, name), trial.params[name])

    def test_t8_sampled_values_match_declared_types(self):
        """
        Optuna must suggest ints for int fields, which can't be done if the Annotated type is not
        correctly unwrapped.
        """
        for cls in (PPFParams, RansacParams, Ransac3DoFParams, VSACSe2Params):
            hints = typing.get_type_hints(cls, include_extras=True)
            p = cls.sample_optuna(optuna.create_study().ask())
            for name in cls.search_space():
                declared = _unwrap_optional(hints[name])
                with self.subTest(cls=cls.__name__, field=name):
                    self.assertIsInstance(getattr(p, name), declared)


if __name__ == "__main__":
    unittest.main()
