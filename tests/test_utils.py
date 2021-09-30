import unittest
import numpy as np

from table_reconstruction.utils import mask_utils, lines_utils, cell_utils, table_utils


class TestMaskUtils(unittest.TestCase):
    def test_normalize(self):
        mask = np.random.random((300, 300, 1))
        img = np.random.random((200, 200, 3))
        resized_mask = mask_utils.normalize(img, mask)
        self.assertEqual(img.shape[0:2], resized_mask.shape[0:2])

    def test_get_lines(self):
        # create a mask with horizontal lines and vertical lines
        #  _________________
        # |                 |
        # |                 |
        # |_________________|
        #
        mask = np.ones((300, 300, 1))
        zeros_part = np.zeros((290, 290, 1))
        mask[5:295, 5:295, :] = zeros_part

        ver_lines_mask = mask_utils.get_ver_lines_mask(mask, 100)
        self.assertTrue(np.array_equal(ver_lines_mask[:, 0:5], np.ones((300, 5))))  # check vertical lines
        self.assertTrue(np.array_equal(ver_lines_mask[:, 295:300], np.ones((300, 5))))  # check vertical lines
        self.assertFalse(np.array_equal(ver_lines_mask[0:5, :], np.ones((5, 300))))  # check no horizontal lines
        self.assertFalse(np.array_equal(ver_lines_mask[295:300, :], np.ones((5, 300))))  # check no horizontal lines

        hor_lines_mask = mask_utils.get_hor_lines_mask(mask, 100)
        self.assertFalse(np.array_equal(hor_lines_mask[:, 0:5], np.ones((300, 5))))  # check no vertical lines
        self.assertFalse(np.array_equal(hor_lines_mask[:, 295:300], np.ones((300, 5))))  # check no vertical lines
        self.assertTrue(np.array_equal(hor_lines_mask[0:5, :], np.ones((5, 300))))  # check horizontal lines
        self.assertTrue(np.array_equal(hor_lines_mask[295:300, :], np.ones((5, 300))))  # check horizontal lines

class TestLinesUtils(unittest.TestCase):
    def setUp(self):
        # create a mask with horizontal lines and vertical lines
        #  _________________
        # |                 |
        # |                 |
        # |_________________|
        #
        mask = np.ones((300, 300, 1))
        zeros_part = np.zeros((290, 290, 1))
        mask[5:295, 5:295, :] = zeros_part
        self.mask_1 = mask
        # create a mask with horizontal lines and vertical lines
        #  __________________
        # |        |         |
        # |________|_________|
        # |        |         |
        # |________|_________|
        #
        mask_2 = mask = np.ones((300, 300, 1))
        mask_2[5:145, 5:145, :] = np.zeros_like(mask_2[5:145, 5:145, :])
        mask_2[150:295, 5:145, :] = np.zeros_like(mask_2[150:295, 5:145, :])
        mask_2[5:145, 150:295, :] = np.zeros_like(mask_2[5:145, 150:295, :])
        mask_2[150:295, 150:295, :] = np.zeros_like(mask_2[150:295, 150:295, :])
        self.mask_2 = mask_2

    def test_is_line(self):
        _, ver_lines_coord, hor_lines_coord = lines_utils.get_coordinates(
            self.mask_1, ths=15
        )
        is_ver_line = lines_utils.is_line([0, 0, 0, 299], ver_lines_coord, 1, 1)
        is_hor_line = lines_utils.is_line([0, 0, 299, 0], hor_lines_coord, 0, 1)
        self.assertTrue(is_ver_line)
        self.assertTrue(is_hor_line)
        not_ver_line = lines_utils.is_line([0, 0, 299, 0], ver_lines_coord, 1, 1)
        not_hor_line = lines_utils.is_line([0, 0, 0, 299], hor_lines_coord, 0, 1)
        self.assertFalse(not_ver_line)
        self.assertFalse(not_hor_line)

    def test_get_coordinates(self):
        tab_coord, ver_lines_coord, hor_lines_coord = lines_utils.get_coordinates(
            self.mask_1, ths=15
        )
        self.assertEqual(tab_coord, [0, 0, 299, 299])
        self.assertEqual(ver_lines_coord.tolist(), [[0, 0, 0, 299], [299, 0, 299, 299]])
        self.assertEqual(hor_lines_coord.tolist(), [[0, 0, 299, 0], [0, 299, 299, 299]])
class TestCellUtils(unittest.TestCase):
    def setUp(self):
        # create a mask with horizontal lines and vertical lines
        #  _________________
        # |                 |
        # |                 |
        # |_________________|
        #
        mask = np.ones((300, 300, 1))
        zeros_part = np.zeros((290, 290, 1))
        mask[5:295, 5:295, :] = zeros_part
        self.mask_1 = mask
        # create a mask with horizontal lines and vertical lines
        #  __________________
        # |        |         |
        # |________|_________|
        # |        |         |
        # |________|_________|
        #
        mask_2 = mask = np.ones((300, 300, 1))
        mask_2[5:145, 5:145, :] = np.zeros_like(mask_2[5:145, 5:145, :])
        mask_2[150:295, 5:145, :] = np.zeros_like(mask_2[150:295, 5:145, :])
        mask_2[5:145, 150:295, :] = np.zeros_like(mask_2[5:145, 150:295, :])
        mask_2[150:295, 150:295, :] = np.zeros_like(mask_2[150:295, 150:295, :])
        self.mask_2 = mask_2

    def test_get_intersection_points(self):
        tab_coord, ver_lines_coord, hor_lines_coord = lines_utils.get_coordinates(
            self.mask_1, ths=15
        )
        intersect_points, fake_intersect_points = cell_utils.get_intersection_points(
            hor_lines_coord, ver_lines_coord, tab_coord
        )

        self.assertEqual(
            intersect_points.tolist(), [[0., 0.], [299., 0.], [0., 299.], [299., 299.]]
        )

    def test_calculate_cell_coord(self):
        tab_coord, ver_lines_coord, hor_lines_coord = lines_utils.get_coordinates(
            self.mask_1, ths=15
        )
        intersect_points, fake_intersect_points = cell_utils.get_intersection_points(
            hor_lines_coord, ver_lines_coord, tab_coord
        )
        cells = cell_utils.calculate_cell_coordinate(
            intersect_points.copy(),
            False,
            15,
            [hor_lines_coord, ver_lines_coord],
        )
        self.assertEqual(cells, [[0, 299, 0, 299]])  # [x1, x2, y1, y2]

    def test_predict_relation(self):
        # use mask_2 which has 4 cells
        tab_coord, ver_lines_coord, hor_lines_coord = lines_utils.get_coordinates(
            self.mask_2, ths=15
        )
        intersect_points, fake_intersect_points = cell_utils.get_intersection_points(
            hor_lines_coord, ver_lines_coord, tab_coord
        )
        cells = cell_utils.calculate_cell_coordinate(
            intersect_points.copy(),
            False,
            15,
            [hor_lines_coord, ver_lines_coord],
        )
        self.assertEqual(len(cells), 4)
        cells = cell_utils.sort_cell(cells=np.array(cells))
        hor_couple_ids, ver_couple_ids = cell_utils.predict_relation(cells)
        self.assertEqual(hor_couple_ids, [[0, 1], [2, 3]])
        self.assertEqual(ver_couple_ids, [[0, 2], [1, 3]])

class TestTableUtils(unittest.TestCase):
    def setUp(self):
        # create a mask with horizontal lines and vertical lines
        #  __________________
        # |        |         |
        # |________|_________|
        # |        |         |
        # |________|_________|
        #
        mask_1 = mask = np.ones((300, 300, 1))
        mask_1[5:145, 5:145, :] = np.zeros_like(mask_1[5:145, 5:145, :])
        mask_1[150:295, 5:145, :] = np.zeros_like(mask_1[150:295, 5:145, :])
        mask_1[5:145, 150:295, :] = np.zeros_like(mask_1[5:145, 150:295, :])
        mask_1[150:295, 150:295, :] = np.zeros_like(mask_1[150:295, 150:295, :])
        self.mask_1 = mask_1
        # add more mask for testing below

    def test_directed_graph(self):
        tab_coord, ver_lines_coord, hor_lines_coord = lines_utils.get_coordinates(
            self.mask_1, ths=15
        )
        intersect_points, fake_intersect_points = cell_utils.get_intersection_points(
            hor_lines_coord, ver_lines_coord, tab_coord
        )
        cells = cell_utils.calculate_cell_coordinate(
            intersect_points.copy(),
            False,
            15,
            [hor_lines_coord, ver_lines_coord],
        )
        fake_cells = cell_utils.calculate_cell_coordinate(
            fake_intersect_points.copy(), 
            True,
            15,
        )
        cells = cell_utils.sort_cell(cells=np.array(cells))
        fake_cells = cell_utils.sort_cell(cells=np.array(fake_cells))
        hor_couple_ids, ver_couple_ids = cell_utils.predict_relation(cells)
        H_Graph = table_utils.DirectedGraph(len(cells))
        H_Graph.add_edges(hor_couple_ids)
        nb_col = H_Graph.findLongestPath()
        V_Graph = table_utils.DirectedGraph(len(cells))
        V_Graph.add_edges(ver_couple_ids)
        nb_row = V_Graph.findLongestPath()
        self.assertEqual(nb_col, 2)
        self.assertEqual(nb_row, 2)
        span_list = table_utils.convertSpanCell2DocxCoord(
            cells, fake_cells, list(range(len(cells))), nb_col
        )
        self.assertEqual(len(span_list), 4)
        self.assertEqual(
            span_list[0], {'id': 0, 'x': [0, 0], 'y': [0, 0]}
        )
        self.assertEqual(
            span_list[1], {'id': 1, 'x': [1, 1], 'y': [0, 0]}
        )
        self.assertEqual(
            span_list[2], {'id': 2, 'x': [0, 0], 'y': [1, 1]}
        )
        self.assertEqual(
            span_list[3], {'id': 3, 'x': [1, 1], 'y': [1, 1]}
        )

if __name__ == "__main__":
    unittest.main()
