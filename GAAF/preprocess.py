from .preprocessing_module import Preprocessor, setup_argparse

args = setup_argparse()
preprocessor = Preprocessor(args)

preprocessor.run_preprocessing()