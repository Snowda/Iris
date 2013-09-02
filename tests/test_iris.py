"""unit tests for iris"""
import unittest
class TestIris(unittest.TestCase):
    def testInsufficientArgs(self):
        foo = 0
        self.failUnlessRaises(ValueError, Sensor, foo)

    def testchange_cal_settings_args(self):
        foo = 0
        self.failUnlessRaises(ValueError, MyClass, foo)

        try:
            self.assertRaises(myExcTwo,self.myClass.getName)
        except Exception as e:
            pass

    def test_span_offset_args(self):
    
    def test_zero_offset_args(self):
    
    def test_bit_offset_args(self):
