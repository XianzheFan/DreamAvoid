import numpy as np
from openpi_client import image_tools, websocket_client_policy

def _random_observation_agilex() -> dict:
    return {'state': np.ones((14,)), 'images': {'cam_high': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), 'cam_left_wrist': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), 'cam_right_wrist': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)}, 'prompt': 'do something'}

def _random_observation_agilex_rtc() -> dict:
    return {'state': np.ones((14,)), 'images': {'cam_high': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), 'cam_left_wrist': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), 'cam_right_wrist': np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)}, 'prompt': 'do something', 'action_prefix': np.ones((4, 14)), 'delay': np.array(4)}

class OpenpiClient:

    def __init__(self, host: str, port: int, mode: str | None=None) -> None:
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.mode = mode

    def _build_observation(self, payload) -> dict:
        images = [payload['top'], payload['left'], payload['right']]
        images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)) for img in images]
        images = [img.transpose(2, 0, 1) for img in images]
        observation = {'state': payload['state'], 'images': {'cam_high': images[0], 'cam_left_wrist': images[1], 'cam_right_wrist': images[2]}, 'prompt': payload['instruction']}
        if 'action_prefix' in payload and payload['action_prefix'] is not None:
            observation['action_prefix'] = payload['action_prefix']
            observation['delay'] = payload['delay']
        if self.mode is not None:
            observation['mode'] = self.mode
        return observation

    def predict_action(self, payload) -> np.ndarray:
        observation = self._build_observation(payload)
        response = self.client.infer(observation)
        return response['actions']

    def predict_action_streaming(self, payload, on_actions_ready=None) -> np.ndarray:
        observation = self._build_observation(payload)
        response = self.client.infer_streaming(observation, on_actions_ready=on_actions_ready)
        if response is not None:
            return response['actions']

    def warmup(self, rtc: bool=False, streaming: bool=False) -> None:
        if streaming:
            self.client.infer_streaming(_random_observation_agilex())
            if rtc:
                self.client.infer_streaming(_random_observation_agilex_rtc())
        else:
            self.client.infer(_random_observation_agilex())
            if rtc:
                self.client.infer(_random_observation_agilex_rtc())
