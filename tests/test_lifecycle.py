"""Model-independent concurrency tests for shared lifecycle infrastructure."""

import sys
import threading
import time
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.core.lifecycle import (  # noqa: E402
    LazyBackendSlot,
    LifecycleCoordinator,
)


class TestLifecycleError(Exception):
    pass


class FakeBackend:
    def __init__(self):
        self.disposals = 0
        self.fail_disposal = False

    def unload(self):
        self.disposals += 1
        if self.fail_disposal:
            raise RuntimeError("disposal failed")


class LifecycleInfrastructureTests(unittest.TestCase):
    def coordinator(self):
        return LifecycleCoordinator(
            lambda: TestLifecycleError("service is unloading")
        )

    def test_concurrent_first_initialization_publishes_exactly_one_backend(self):
        created = []
        factory_entered = threading.Event()
        release_factory = threading.Event()

        def factory():
            factory_entered.set()
            release_factory.wait(timeout=2)
            backend = FakeBackend()
            created.append(backend)
            return backend

        slot = LazyBackendSlot(factory, lambda backend: backend.unload())
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(slot.get()))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(factory_entered.wait(timeout=1))
        release_factory.set()
        for thread in threads:
            thread.join(timeout=1)

        self.assertEqual(len(created), 1)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(result is created[0] for result in results))

    def test_backend_creation_failure_is_not_published_and_can_retry(self):
        attempts = 0

        def factory():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("creation failed")
            return FakeBackend()

        slot = LazyBackendSlot(factory, lambda backend: backend.unload())
        with self.assertRaises(RuntimeError):
            slot.get()
        self.assertFalse(slot.is_initialized)
        self.assertIsInstance(slot.get(), FakeBackend)
        self.assertEqual(attempts, 2)

    def test_unload_waits_for_inference_and_rejects_new_work(self):
        coordinator = self.coordinator()
        inference_started = threading.Event()
        release_inference = threading.Event()
        cleanup_called = threading.Event()

        def inference():
            with coordinator.operation():
                inference_started.set()
                release_inference.wait(timeout=2)

        worker = threading.Thread(target=inference)
        worker.start()
        self.assertTrue(inference_started.wait(timeout=1))
        unloading = threading.Thread(
            target=lambda: coordinator.unload(cleanup_called.set)
        )
        unloading.start()
        for _ in range(100):
            if coordinator.is_draining:
                break
            time.sleep(0.005)
        self.assertTrue(coordinator.is_draining)
        self.assertFalse(cleanup_called.is_set())
        with self.assertRaises(TestLifecycleError):
            with coordinator.operation():
                pass

        release_inference.set()
        worker.join(timeout=1)
        unloading.join(timeout=1)
        self.assertTrue(cleanup_called.is_set())
        self.assertEqual(coordinator.active_operations, 0)

    def test_inference_failure_restores_active_count(self):
        coordinator = self.coordinator()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with coordinator.operation():
                raise RuntimeError("inference failed")
        self.assertEqual(coordinator.active_operations, 0)
        coordinator.unload(lambda: None)

    def test_repeated_unload_is_safe(self):
        backend = FakeBackend()
        slot = LazyBackendSlot(
            lambda: backend, lambda resource: resource.unload(), backend
        )
        coordinator = self.coordinator()
        coordinator.unload(slot.unload)
        coordinator.unload(slot.unload)
        self.assertEqual(backend.disposals, 1)
        self.assertFalse(slot.is_initialized)

    def test_unload_allows_lazy_reload(self):
        created = []

        def factory():
            backend = FakeBackend()
            created.append(backend)
            return backend

        slot = LazyBackendSlot(factory, lambda backend: backend.unload())
        coordinator = self.coordinator()
        first = slot.get()
        coordinator.unload(slot.unload)
        second = slot.get()
        self.assertIsNot(first, second)
        self.assertEqual(len(created), 2)

    def test_unload_from_active_operation_is_rejected_without_deadlock(self):
        coordinator = self.coordinator()
        with coordinator.operation():
            with self.assertRaises(TestLifecycleError):
                coordinator.unload(lambda: None)
        self.assertEqual(coordinator.active_operations, 0)

    def test_concurrent_unloads_are_serialized_without_admission_window(self):
        coordinator = self.coordinator()
        first_cleanup_started = threading.Event()
        release_first = threading.Event()
        cleanup_lock = threading.Lock()
        concurrent_cleanups = 0
        maximum_concurrent = 0
        cleanup_calls = 0

        def cleanup():
            nonlocal concurrent_cleanups, maximum_concurrent, cleanup_calls
            with cleanup_lock:
                concurrent_cleanups += 1
                cleanup_calls += 1
                maximum_concurrent = max(maximum_concurrent, concurrent_cleanups)
                call_number = cleanup_calls
            if call_number == 1:
                first_cleanup_started.set()
                release_first.wait(timeout=2)
            with cleanup_lock:
                concurrent_cleanups -= 1

        first = threading.Thread(target=lambda: coordinator.unload(cleanup))
        second = threading.Thread(target=lambda: coordinator.unload(cleanup))
        first.start()
        self.assertTrue(first_cleanup_started.wait(timeout=1))
        second.start()
        time.sleep(0.05)
        with self.assertRaises(TestLifecycleError):
            with coordinator.operation():
                pass
        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)

        self.assertEqual(cleanup_calls, 2)
        self.assertEqual(maximum_concurrent, 1)
        self.assertFalse(coordinator.is_draining)

    def test_backend_disposal_failure_leaves_resource_retryable(self):
        backend = FakeBackend()
        backend.fail_disposal = True
        slot = LazyBackendSlot(
            lambda: backend, lambda resource: resource.unload(), backend
        )
        coordinator = self.coordinator()
        with self.assertRaisesRegex(RuntimeError, "disposal failed"):
            coordinator.unload(slot.unload)
        self.assertTrue(slot.is_initialized)
        self.assertFalse(coordinator.is_draining)

        backend.fail_disposal = False
        coordinator.unload(slot.unload)
        self.assertFalse(slot.is_initialized)
        self.assertEqual(backend.disposals, 2)

    def test_unload_after_failed_load_is_safe(self):
        slot = LazyBackendSlot(
            lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
            lambda backend: backend.unload(),
        )
        with self.assertRaises(RuntimeError):
            slot.get()
        self.coordinator().unload(slot.unload)
        self.assertFalse(slot.is_initialized)

    def test_translation_style_dual_slots_remain_independent(self):
        en_indic = FakeBackend()
        indic_en = FakeBackend()
        en_slot = LazyBackendSlot(
            lambda: en_indic, lambda backend: backend.unload()
        )
        indic_slot = LazyBackendSlot(
            lambda: indic_en, lambda backend: backend.unload()
        )
        coordinator = self.coordinator()

        with coordinator.operation():
            self.assertIs(en_slot.get(), en_indic)
            self.assertIs(indic_slot.get(), indic_en)
        coordinator.unload(lambda: (en_slot.unload(), indic_slot.unload()))

        self.assertEqual(en_indic.disposals, 1)
        self.assertEqual(indic_en.disposals, 1)
        self.assertFalse(en_slot.is_initialized)
        self.assertFalse(indic_slot.is_initialized)


if __name__ == "__main__":
    unittest.main()
