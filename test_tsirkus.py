from tsirkus import Tsirkus 
import unittest
import numpy as np
import pdb;


class TestTsirkus(unittest.TestCase):
    def test_default(self):
        t = Tsirkus()
        assert t.N == t.shape[0]*t.shape[1]

    def test_P_property_nojumps(self):
        t = Tsirkus(jumps=[])
        np.testing.assert_array_almost_equal(np.sum(t.P,axis=1),np.ones(t.N))

    def test_P_property(self):
        t = Tsirkus()
        np.testing.assert_array_almost_equal(np.sum(t.P[np.delete(np.arange(t.N),t.jumps[0])],axis=1),np.ones(t.N - len(t.jumps[0]))) # check no-jump rows sum to 1
        np.testing.assert_array_almost_equal(np.sum(t.P[t.jumps[0]],axis=1),np.zeros(len(t.jumps[0]))) # check jump rows sum to 0

    def test_default_nojumps(self):
        t = Tsirkus(jumps = [])
        np.testing.assert_array_equal(t.path,np.arange(t.shape[0]*t.shape[1]))

    def test_10x1(self):
        t = Tsirkus(jumps = [], shape=(10,1))

    def test_evolve_final(self):
        t = Tsirkus()
        p0 = np.eye(t.N,1)
        p_final = 0.5
        for i, p in enumerate(t.evolve_final(p0, p_final)):
            np.testing.assert_allclose(np.sum(p),1.)
            print(i,p[-1])
        assert p[-1] >= p_final 

    def test_evolve_final_default(self):
        t = Tsirkus()
        p0 = np.eye(t.N,1)
        p_final = 0.5
        for i, pi in enumerate(t.evolve_final(p0, p_final)):
            pass
        assert pi[-1] >= p_final 
        for j, pj in enumerate(t.evolve_final()):
            if i == j:
                break
        np.testing.assert_allclose(pi,pj)

    def test_evolve_turns(self):
        t = Tsirkus()
        p0 = np.eye(t.N,1)
        p_final = 0.5
        for i, pi in enumerate(t.evolve_final(p0, p_final)):
            pass
        assert pi[-1] >= p_final 
        np.testing.assert_allclose(t.evolve_turns(p0,i),pi)

if __name__ == '__main__':
    unittest.main()

