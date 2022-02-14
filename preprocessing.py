from preprocessing.preprocess_Locator_data import Preprocessor, setup_argparse

args = setup_argparse()
preprocessor = Preprocessor(args)

preprocessor.run_preprocessing()