def enable_interceptor(self, interceptor_name: str | None = None) -> bool:
    """Enable memory recording (legacy method)"""
    # Only LiteLLM native callbacks supported (interceptor_name ignored)
    results: dict[str, bool] = self.memory_manager.enable(["litellm_native"])
    return results.get("success", False)


def disable_interceptor(self, interceptor_name: str | None = None) -> bool:
    """Disable memory recording (legacy method)"""
    # Only LiteLLM native callbacks supported (interceptor_name ignored)
    results: dict[str, bool] = self.memory_manager.disable()
    return results.get("success", False)
