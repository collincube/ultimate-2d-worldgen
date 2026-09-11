"""Offline regression checks for the standalone simulation's public API."""
import json
import unittest

from infernal_data_first import (InfernalWorld, WorldConfig, Terrain, Ground,
                                 TERRAIN_NAME, GROUND_NAME, DAILY_CHALLENGES)


class SimulationTests(unittest.TestCase):
    def make_world(self, seed=1337):
        return InfernalWorld(WorldConfig(width=24, height=24, seed=seed,
                                        challenge_date='2026-09-10'))

    def test_seeded_ticks_are_reproducible(self):
        first, second = self.make_world(), self.make_world()
        first.step(2)
        second.step(2)
        self.assertEqual(first.state.tick, 2)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertNotEqual(first.state.terrain, self.make_world(42).state.terrain)

    def test_json_round_trip_preserves_snapshot(self):
        world = self.make_world()
        world.step(2)
        restored = InfernalWorld.from_json(world.to_json())
        self.assertEqual(json.loads(world.to_json()), json.loads(restored.to_json()))

    def test_share_code_preserves_snapshot(self):
        world = self.make_world()
        world.step(1)
        restored = InfernalWorld.from_share_code(world.to_share_code())
        self.assertEqual(json.loads(world.to_json()), json.loads(restored.to_json()))

    def test_renamed_tiles_keep_numeric_snapshot_ids(self):
        world = self.make_world()
        # Numeric IDs are the persisted format; terminology changes keep them stable.
        self.assertEqual(int(Terrain.ALIEN_BIOMASS), 6)
        self.assertEqual(int(Ground.EMBER_RUNE), 2)
        world.state.terrain[0] = 6
        world.apply_tool(0, 0, 'ember_rune')
        self.assertEqual(world.state.ground[0], 2)
        self.assertGreaterEqual(world.compute_metrics().counts['ember_rune'], 1)
        restored = InfernalWorld.from_json(world.to_json())
        self.assertEqual(TERRAIN_NAME[Terrain(restored.state.terrain[0])], 'alien_biomass')
        self.assertEqual(GROUND_NAME[Ground(restored.state.ground[0])], 'ember_rune')
        self.assertEqual(DAILY_CHALLENGES[0].title, 'Lava Moat')


if __name__ == '__main__':
    unittest.main()
