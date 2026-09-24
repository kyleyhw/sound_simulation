"""Observable targets on toy rooms (plan 6.1)."""

from __future__ import annotations

import numpy as np

from acoustic_system.imaging.targets import (
    free_space_outline,
    illuminated_boundary,
    obstacle_polygons,
    rasterize_polygons,
    signed_distance,
    surface_cells,
)


def _room() -> np.ndarray:
    m = np.zeros((32, 32), dtype=bool)
    m[10:16, 10:16] = True  # front block
    m[11:15, 20:23] = True  # block hidden behind it (seen from the left)
    return m


SRC = np.array([[13, 3]])
MICS = np.array([[[12, 3], [14, 4]]])


def test_line_of_sight_lights_only_the_facing_surface():
    m = _room()
    ill = illuminated_boundary(m, SRC, MICS, first_order=False)
    assert ill[10:16, 10].all()  # the face towards the device
    assert not ill[10:16, 15].any()  # the back face
    assert not ill[11:15, 20:23].any()  # the shadowed block
    assert not (ill & ~surface_cells(m)).any()  # only surface cells, never interiors
    assert not ill[11:15, 11:15].any()


def test_first_order_wall_bounces_reach_side_faces():
    m = _room()
    direct = illuminated_boundary(m, SRC, MICS, first_order=False)
    bounce = illuminated_boundary(m, SRC, MICS, first_order=True)
    assert (bounce >= direct).all()
    # Via the i = 0 and i = 31 walls the top and bottom faces become visible.
    assert bounce[10, 11:15].any() and bounce[15, 11:15].any()
    # "any device" mode is at least as permissive as the bistatic "pose" mode.
    anyd = illuminated_boundary(m, SRC, MICS, mode="any")
    assert (anyd >= bounce).all()


def test_polygons_round_trip_and_outline():
    m = _room()
    polys = obstacle_polygons(m)
    assert len(polys) == 2
    assert all(4 <= len(p) <= 8 for p in polys)  # a rectangle simplifies to a few vertices
    np.testing.assert_array_equal(rasterize_polygons(polys, m.shape), m)
    region, outline = free_space_outline(m, [(13, 3)])
    assert region[13, 3] and not region[m].any()
    assert not region[0].any() and not region[:, -1].any()  # outer p = 0 walls
    # Outer outline plus two holes (the blocks); rasterising gives the free region.
    assert len(outline) == 3
    np.testing.assert_array_equal(rasterize_polygons(outline, m.shape), region)


def test_signed_distance_field():
    m = np.zeros((20, 20), dtype=bool)
    m[8:12, 8:12] = True
    phi = signed_distance(m)
    assert phi[10, 5] == 2.5  # three cells from the face at column 8
    assert phi[10, 8] == -0.5 and phi[10, 7] == 0.5  # zero level set on the face
    assert phi[m].max() < 0 < phi[~m].min()
    assert signed_distance(m, truncate=1.0).max() == 1.0
    walls = signed_distance(np.zeros((20, 20), dtype=bool), include_walls=True)
    assert walls[10, 1] == 0.5 and walls[10, 10] == 8.5
    assert np.isinf(signed_distance(np.zeros((5, 5), dtype=bool))).all()
