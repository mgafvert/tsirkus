from tsirkus import Tsirkus 
import unittest
import pdb;


class TestTsirkus(unittest.TestCase):
    def test_default(self):
        t = Tsirkus()
        assert t.N == t.shape[0]*t.shape[1]
    def test_10x1(self):
        t = Tsirkus(jumps = [], shape=(10,1))
        

if __name__ == '__main__':
    unittest.main()

