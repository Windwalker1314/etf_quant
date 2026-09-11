import argparse

from steadyquant.sector_data import sync_sector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    sync_sector(pilot=parser.parse_args().pilot)
