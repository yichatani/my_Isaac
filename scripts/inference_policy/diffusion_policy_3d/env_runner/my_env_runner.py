# diffusion_policy_3d/env_runner/dummy_runner.py

from diffusion_policy_3d.env_runner.base_runner import BaseRunner

class DummyRunner(BaseRunner):
    def run(self, policy, *args, **kwargs):
        return {}
