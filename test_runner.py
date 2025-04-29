import unittest
import sys
import os

# Import test suites
from test_suite import (
    TestReplayBuffer,
    TestDQN,
    TestDQNAgent,
    TestHSREnvironment
)

from mock_test_suite import (
    TestHSREnvironmentMocking,
    TestEnvironmentEndToEnd
)

from integration_test_suite import (
    TestTrainingScript,
    TestMemoryManagement,
    TestAgentPerformance
)

def run_all_tests():
    """Run all test suites."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes from the basic test suite
    suite.addTest(unittest.makeSuite(TestReplayBuffer))
    suite.addTest(unittest.makeSuite(TestDQN))
    suite.addTest(unittest.makeSuite(TestDQNAgent))
    suite.addTest(unittest.makeSuite(TestHSREnvironment))
    
    # Add test classes from the mock test suite
    suite.addTest(unittest.makeSuite(TestHSREnvironmentMocking))
    suite.addTest(unittest.makeSuite(TestEnvironmentEndToEnd))
    
    # Add test classes from the integration test suite
    suite.addTest(unittest.makeSuite(TestTrainingScript))
    suite.addTest(unittest.makeSuite(TestMemoryManagement))
    suite.addTest(unittest.makeSuite(TestAgentPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_specific_suite(suite_name):
    """Run a specific test suite."""
    suite = unittest.TestSuite()
    
    if suite_name == "basic":
        suite.addTest(unittest.makeSuite(TestReplayBuffer))
        suite.addTest(unittest.makeSuite(TestDQN))
        suite.addTest(unittest.makeSuite(TestDQNAgent))
        suite.addTest(unittest.makeSuite(TestHSREnvironment))
    elif suite_name == "mock":
        suite.addTest(unittest.makeSuite(TestHSREnvironmentMocking))
        suite.addTest(unittest.makeSuite(TestEnvironmentEndToEnd))
    elif suite_name == "integration":
        suite.addTest(unittest.makeSuite(TestTrainingScript))
        suite.addTest(unittest.makeSuite(TestMemoryManagement))
        suite.addTest(unittest.makeSuite(TestAgentPerformance))
    else:
        print(f"Unknown test suite: {suite_name}")
        print("Available suites: basic, mock, integration")
        return None
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        suite_name = sys.argv[1]
        result = run_specific_suite(suite_name)
    else:
        # Run all tests
        result = run_all_tests()
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())