"""Reproduce the bounded family strategy comparison without changing live config."""

from steadyquant.family_optimization import run_family_optimization

if __name__ == "__main__":
    print(run_family_optimization())
