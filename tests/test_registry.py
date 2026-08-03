"""Model-independent tests for ServiceRegistry ownership and shutdown."""

import asyncio
import sys
import threading
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.core.registry import ServiceRegistry  # noqa: E402


class FakeService:
    def __init__(self, fail_unload=False):
        self.unload_calls = 0
        self.fail_unload = fail_unload

    def unload(self):
        self.unload_calls += 1
        if self.fail_unload:
            raise RuntimeError("unload failed")


class ServiceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ServiceRegistry()

    def test_status_contract_is_preserved(self):
        self.assertEqual(
            self.registry.status(),
            {
                "translation": False,
                "ocr": False,
                "stt": False,
                "tts": False,
                "lang_detect": False,
            },
        )

    def test_lazy_getter_publishes_one_shared_instance(self):
        created = []

        def factory():
            service = FakeService()
            created.append(service)
            return service

        self.registry._factories["translation"] = factory

        async def get_concurrently():
            return await asyncio.gather(
                *(self.registry.get_translation_service() for _ in range(20))
            )

        results = asyncio.run(get_concurrently())
        self.assertEqual(len(created), 1)
        self.assertTrue(all(result is created[0] for result in results))
        self.assertTrue(self.registry.status()["translation"])

    def test_factory_failure_is_retryable(self):
        attempts = 0

        def factory():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("creation failed")
            return FakeService()

        self.registry._factories["ocr"] = factory
        with self.assertRaisesRegex(RuntimeError, "creation failed"):
            asyncio.run(self.registry.get_ocr_service())
        self.assertFalse(self.registry.status()["ocr"])
        self.assertIsInstance(
            asyncio.run(self.registry.get_ocr_service()), FakeService
        )
        self.assertEqual(attempts, 2)

    def test_unload_delegates_and_clears_all_references(self):
        services = {
            name: FakeService()
            for name in ("translation", "ocr", "stt", "tts")
        }
        self.registry._services.update(services)
        detector = object()
        self.registry._services["lang_detector"] = detector

        self.registry.unload_all()

        self.assertTrue(
            all(service.unload_calls == 1 for service in services.values())
        )
        self.assertTrue(
            all(loaded is False for loaded in self.registry.status().values())
        )

    def test_shutdown_continues_after_one_service_fails(self):
        failing = FakeService(fail_unload=True)
        healthy = FakeService()
        self.registry._services["ocr"] = failing
        self.registry._services["translation"] = healthy

        with self.assertLogs("ai_translator_api.core.registry", level="ERROR"):
            self.registry.unload_all()

        self.assertEqual(failing.unload_calls, 1)
        self.assertEqual(healthy.unload_calls, 1)
        self.assertFalse(any(self.registry.status().values()))

    def test_singleton_creation_is_thread_safe(self):
        original = ServiceRegistry._instance
        ServiceRegistry._instance = None
        try:
            instances = []
            threads = [
                threading.Thread(
                    target=lambda: instances.append(
                        ServiceRegistry.get_instance()
                    )
                )
                for _ in range(20)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=1)
            self.assertEqual(len(instances), 20)
            self.assertTrue(
                all(instance is instances[0] for instance in instances)
            )
        finally:
            ServiceRegistry._instance = original


if __name__ == "__main__":
    unittest.main()
