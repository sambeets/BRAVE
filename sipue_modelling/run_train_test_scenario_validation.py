"""Design on uncertainty_id 1-7, evaluate on 8-10, per archetype.

Writes output/validation/train-test/train_test_1_7_vs_rest.csv.
"""
from brave.config import Config
from brave.runners import run_train_test_scenario_validation

if __name__ == "__main__":
    cfg = Config()
    cfg.export_csv = True
    cfg.export_plots = False
    cfg.export_regret_landscape = False
    cfg.train_test_max_train_id = 7
    run_train_test_scenario_validation(cfg)
