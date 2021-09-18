import unittest


class TestModule(unittest.TestCase):
    def test_module_exist(self):
        exception = None
        try:
            import table_reconstruction

            print(table_reconstruction.__version__)
        except Exception as e:
            exception = e
        self.assertIsNone(exception)


if __name__ == "__main__":
    unittest.main()
