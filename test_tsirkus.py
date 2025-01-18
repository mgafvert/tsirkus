from tsirkus import Tsirkus 
import unittest


class TestTsirkus(unittest.TestCase):
    def test_tsirkus(self):
        t = Tsirkus()
        assert t.N == t.shape[0]*t.shape[1]
    


if __name__ == '__main__':
    unittest.main()

