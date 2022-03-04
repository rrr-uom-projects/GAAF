from GAAF.training_module import train_locator, setup_argparse

args = setup_argparse()
train_locator = train_locator(args)

train_locator.run_training()