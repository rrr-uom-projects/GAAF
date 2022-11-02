from .training_module import test_locator, setup_argparse

args = setup_argparse()
test_locator = test_locator(args)

test_locator.run_testing()