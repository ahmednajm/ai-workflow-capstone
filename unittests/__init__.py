
import unittest

## api tests
from unittests.client_side_ApiTests import *
ApiTestSuite = unittest.TestLoader().loadTestsFromTestCase(TestApi)

## model tests
from unittests.server_side_ModelTests import *
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(Test_Model_Module)

## logger tests
from unittests.server_side_LoggerTests import *
LoggerTestSuite = unittest.TestLoader().loadTestsFromTestCase(Test_Logger_Module)

MainSuite = unittest.TestSuite([LoggerTestSuite,ModelTestSuite, ApiTestSuite])