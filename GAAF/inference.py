from inference_module import Locator_inference_module, setup_argparse

args = setup_argparse()
inference_module = Locator_inference_module(args)

inference_module.run_inference()