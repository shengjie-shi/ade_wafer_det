"""行为测试：检查已知缺陷的检出与正常图误报，而非复刻背景拟合公式。"""

from dataclasses import replace
import unittest

import numpy as np

from wafer_defects import DetectorConfig, WaferGeometry, detect
from wafer_defects.synthetic import PROFILES, SyntheticSpot, make_wafer


class DetectorBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 一组样例同时用于正、负样本测试，随机种子固定，结果可复现。
        cls.samples = {name: make_wafer(profile=name) for name in PROFILES}
        cls.positive = {name: detect(sample.image) for name, sample in cls.samples.items()}
        cls.negative = {name: detect(sample.normal_image) for name, sample in cls.samples.items()}

    def assert_spots_found(self, result, spots, tolerance=2.5):
        """每个真值位置都须匹配独立候选，位置误差小且亮暗极性正确。"""
        matched = set()
        for spot in spots:
            polarity = "bright" if spot.amplitude > 0 else "dark"
            candidates = [defect for defect in result.defects
                          if defect.polarity == polarity and defect.id not in matched
                          and np.hypot(defect.centroid_xy[0] - spot.x,
                                       defect.centroid_xy[1] - spot.y) <= tolerance]
            self.assertTrue(candidates, f"漏检：{spot}；实际候选：{result.defects}")
            matched.add(min(candidates, key=lambda d: np.hypot(
                d.centroid_xy[0] - spot.x, d.centroid_xy[1] - spot.y)).id)
        return matched

    def test_different_ring_counts_and_widths_detect_all_injected_spots(self):
        for name, sample in self.samples.items():
            with self.subTest(profile=name):
                result = self.positive[name]
                self.assertEqual(len(result.defects), 3, "应检出三处注入点，且无多余候选")
                self.assert_spots_found(result, sample.spots)

    def test_normal_transitions_are_not_reported(self):
        for name, result in self.negative.items():
            with self.subTest(profile=name):
                self.assertEqual(result.defects, [], "正常图的环形交界不应产生候选")
                self.assertFalse(result.labels.any())

    def test_noise_free_unequal_width_rings_are_not_defects(self):
        sample = make_wafer(profile="uneven_widths", noise_std=0, spots=())
        result = detect(sample.image)
        self.assertEqual(result.defects, [])
        # 独立已知背景用于检查拟合是否保留了真实明暗变化。
        error = np.abs(result.background[result.analysis_mask]
                       - sample.true_background[result.analysis_mask])
        self.assertLess(float(np.quantile(error, .99)), 4.0)

    def test_additional_noise_seeds_on_clean_images(self):
        for seed in (7, 91, 2026):
            with self.subTest(seed=seed):
                sample = make_wafer(profile="many_rings", noise_std=2.5, seed=seed, spots=())
                self.assertEqual(detect(sample.image).defects, [])

    def test_weak_bright_and_dark_spots_at_normal_transition(self):
        reference = self.samples["uneven_widths"].spots[2]
        for amplitude in (18, -18):
            with self.subTest(amplitude=amplitude):
                spot = replace(reference, amplitude=amplitude)
                sample = make_wafer(profile="uneven_widths", noise_std=2,
                                    spots=(spot,), seed=101)
                result = detect(sample.image)
                self.assertEqual(len(result.defects), 1)
                self.assert_spots_found(result, sample.spots)

    def test_three_and_seven_pixel_spots(self):
        spots = (SyntheticSpot(110, 130, 35, 3), SyntheticSpot(240, 230, -24, 7))
        sample = make_wafer(profile="few_rings", spots=spots)
        result = detect(sample.image)
        self.assertEqual(len(result.defects), 2)
        self.assert_spots_found(result, spots)

    def test_defect_straddling_polar_angle_seam(self):
        # y 与环心一致，5×5 点跨越 0°/360°，应为一处完整候选。
        spot = SyntheticSpot(280, 182, 30)
        sample = make_wafer(profile="many_rings", ring_center_offset_xy=(0, 0),
                            ellipticity=0, spots=(spot,))
        result = detect(sample.image)
        self.assertEqual(len(result.defects), 1)
        self.assert_spots_found(result, (spot,))

    def test_subpixel_ring_center_can_differ_from_wafer_center(self):
        sample = make_wafer(profile="many_rings", ring_center_offset_xy=(2.25, -1.75),
                            ellipticity=0, illumination_xy=(0, 0))
        result = detect(sample.image)
        error = np.hypot(result.ring_center_xy[0] - sample.ring_center_xy[0],
                          result.ring_center_xy[1] - sample.ring_center_xy[1])
        self.assertLess(error, .45)
        self.assertEqual(len(result.defects), 3)
        self.assert_spots_found(result, sample.spots)

    def test_manual_geometry_is_supported(self):
        sample = make_wafer(profile="few_rings", ring_center_offset_xy=(0, 0))
        config = DetectorConfig(refine_center=False)
        result = detect(sample.image, config, geometry=sample.geometry)
        self.assertEqual(result.wafer_geometry, sample.geometry)
        self.assertEqual(result.ring_center_xy, (sample.geometry.center_x, sample.geometry.center_y))
        self.assertEqual(len(result.defects), 3)
        self.assert_spots_found(result, sample.spots)

    def test_equivalent_bgr_image_has_same_detections(self):
        sample = self.samples["many_rings"]
        bgr = np.repeat(sample.image[:, :, None], 3, axis=2)
        result = detect(bgr)
        np.testing.assert_array_equal(result.labels, self.positive["many_rings"].labels)

    def test_bgra_image_is_supported(self):
        sample = self.samples["single_ring"]
        bgra = np.dstack((sample.image, sample.image, sample.image,
                          np.full_like(sample.image, 255)))
        result = detect(bgra)
        self.assertEqual(len(result.defects), 3)
        self.assert_spots_found(result, sample.spots)

    def test_uniform_brightness_offset_does_not_change_candidate_count(self):
        sample = self.samples["few_rings"]
        shifted = sample.image.astype(np.float32) + 20
        result = detect(shifted)
        self.assertEqual(len(result.defects), 3)
        self.assert_spots_found(result, sample.spots)

    def test_rotating_image_preserves_spot_detection(self):
        sample = self.samples["uneven_widths"]
        result = detect(np.rot90(sample.image))
        expected = tuple(SyntheticSpot(spot.y, sample.image.shape[1] - 1 - spot.x,
                                       spot.amplitude, spot.size) for spot in sample.spots)
        self.assertEqual(len(result.defects), 3)
        self.assert_spots_found(result, expected)

    def test_custom_mask_hole_remains_uninspected(self):
        sample = self.samples["many_rings"]
        yy, xx = np.indices(sample.image.shape)
        mask = np.hypot(xx - sample.geometry.center_x, yy - sample.geometry.center_y) <= sample.geometry.radius
        hidden = sample.spots[0]
        mask[hidden.y - 8:hidden.y + 9, hidden.x - 8:hidden.x + 9] = False
        result = detect(sample.image, geometry=sample.geometry, wafer_mask=mask)
        self.assertFalse(result.analysis_mask[hidden.y, hidden.x])
        self.assertEqual(len(result.defects), 2)
        self.assert_spots_found(result, sample.spots[1:])

    def test_isolated_hot_pixel_is_rejected_by_minimum_area(self):
        sample = make_wafer(profile="single_ring", spots=(SyntheticSpot(110, 140, 80, 1),))
        self.assertEqual(detect(sample.image).defects, [])

    def test_exclusions_and_intermediate_arrays_are_explicit(self):
        sample = self.samples["few_rings"]
        result = self.positive["few_rings"]
        self.assertFalse(result.analysis_mask[0, 0])
        self.assertFalse(result.analysis_mask[round(result.ring_center_xy[1]), round(result.ring_center_xy[0])])
        for array in (result.background, result.residual, result.noise_map, result.score_map):
            self.assertEqual(array.shape, (365, 361))
            self.assertTrue(np.isfinite(array).all())
            self.assertTrue((array[~result.analysis_mask] == 0).all())
        self.assertTrue((result.labels[~result.analysis_mask] == 0).all())
        for defect in result.defects:
            self.assertEqual(int((result.labels == defect.id).sum()), defect.area)
        self.assertEqual(int((sample.truth_labels > 0).sum()), 75)

    def test_detector_does_not_modify_input(self):
        sample = self.samples["few_rings"]
        original = sample.image.copy()
        detect(sample.image)
        np.testing.assert_array_equal(sample.image, original)


class ValidationTests(unittest.TestCase):
    def test_invalid_images_fail_clearly(self):
        cases = (np.zeros((10, 10)), np.zeros((40, 40, 2)), np.full((40, 40), np.nan),
                 np.full((40, 40), np.inf), np.full((40, 40), -1),
                 np.full((40, 40), 1000, dtype=np.uint16), np.zeros((40, 40)))
        for image in cases:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaises(ValueError):
                    detect(image)

    def test_invalid_configuration_fails_early(self):
        cases = ({"radial_step_px": 0}, {"radial_step_px": 2}, {"noise_floor": float("nan")},
                 {"scales": (2, 4)}, {"scales": ()}, {"scales": (3, 3)},
                 {"min_area": 0}, {"max_area": 1}, {"angular_samples": -1},
                 {"edge_margin_px": -1}, {"background_iterations": 0}, {"min_area": 2.5})
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    DetectorConfig(**config)

    def test_invalid_masks_and_geometry_fail(self):
        image = make_wafer(spots=()).image
        for mask in (np.ones((12, 12)), np.zeros(image.shape), np.full(image.shape, np.nan)):
            with self.subTest(mask_shape=mask.shape):
                with self.assertRaises(ValueError):
                    detect(image, wafer_mask=mask)
        for geometry in (WaferGeometry(180, 182, -2), WaferGeometry(-1, 182, 170),
                         WaferGeometry(float("nan"), 182, 170)):
            with self.subTest(geometry=geometry):
                with self.assertRaises(ValueError):
                    detect(image, geometry=geometry)


if __name__ == "__main__":
    unittest.main()
