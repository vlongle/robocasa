import cv2
import numpy as np

from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv


class _CameraKeyConverter:
    @staticmethod
    def map_obs(raw_obs):
        del raw_obs
        return {}

    @staticmethod
    def get_camera_config():
        return (["video.res256_image_side_0"], ["sideview"], None, None)


def test_512_render_keeps_distinct_256_policy_and_512_guidance_images():
    env = object.__new__(GrootRoboCasaEnv)
    env.key_converter = _CameraKeyConverter()
    env.camera_widths = 512

    rows, cols = np.indices((512, 512))
    raw_image = np.stack((rows % 256, cols % 256, (rows + cols) % 256), axis=-1).astype(np.uint8)
    observation = env.get_groot_observation(
        {"sideview_image": raw_image, "language": "turn the sink spout"}
    )

    policy_image = observation["video.res256_image_side_0"]
    guidance_image = observation["video.res512_image_side_0"]
    assert policy_image.shape == (256, 256, 3)
    assert guidance_image.shape == (512, 512, 3)
    np.testing.assert_array_equal(
        policy_image,
        cv2.resize(raw_image, (256, 256), interpolation=cv2.INTER_AREA),
    )
    np.testing.assert_array_equal(guidance_image, raw_image)
    assert not np.shares_memory(guidance_image, raw_image)
