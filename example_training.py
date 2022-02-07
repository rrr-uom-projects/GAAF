from training.train_locator import train_locator, setup_argparse

args = setup_argparse()
train_locator = train_locator(args)

train_locator.run_training()